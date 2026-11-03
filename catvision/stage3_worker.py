"""One resident GPU consumer; independent routes remain parallel inside each job."""
import json
import logging
import os
from pathlib import Path
import threading
import traceback

import pika

from .common import ROOT, load_image
from .stage3_queue import declare
from .stage3_settings import Settings
from .stage3_store import Store

log = logging.getLogger('catvision.worker')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def process_job(store, identity, runner):
    job = store.claim(identity)
    if job is None:
        return False
    directory = store.settings.data_dir/identity
    directory.mkdir(parents=True,exist_ok=True)

    def progress(message):
        store.change(identity,message=message)

    try:
        image_id = job['parent_id'] if job['kind']=='image_question' else identity
        image = load_image(store.settings.data_dir/image_id/'input.png',max_side=1600)
        if job['kind']=='image_question':
            progress('正在观察原图并回答问题')
            result = runner.answer_question(image,job['question'])
            message = '回答完成'
        else:
            result = runner(image,directory,progress)
            message = '识别完成'
        outcome = {**job,'state':'completed','message':message,'result':result}
        atomic_json(directory/'result.json',outcome)
        store.finish(identity,result,message)
        log.info('Task completed: %s',identity)
        return True
    except Exception as exc:
        error = traceback.format_exc()
        (directory/'error.log').write_text(error,encoding='utf-8')
        message = f'任务失败：{type(exc).__name__}: {exc}；工作端已停止，请查看日志后手动恢复'
        store.change(identity,state='error',message=message)
        atomic_json(directory/'result.json',{**job,'state':'error','message':message})
        log.exception('Task failed; stopping worker: %s',identity)
        raise


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CATVISION_FAST_INFERENCE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    settings = Settings()
    store = Store(settings)
    from .run_control import GpuReservation
    from .server import LocalComparison, model_files_ready
    stop = threading.Event()
    runner = LocalComparison(keep_loaded=True)
    connection = None

    def heartbeat():
        while not stop.wait(5):
            try:
                store.worker(True,'模型已就绪')
            except Exception:
                log.exception('Worker heartbeat could not reach database')

    with GpuReservation(ROOT):
        store.interrupt_running()
        store.worker(False,'正在加载两个识别模型')
        try:
            if not model_files_ready(ROOT):
                raise RuntimeError('模型文件不完整，请按部署说明恢复权重')
            runner.preload()
            store.worker(True,'模型已就绪')
            thread = threading.Thread(target=heartbeat,name='worker-heartbeat',daemon=True)
            thread.start()
            parameters = pika.URLParameters(settings.amqp_url)
            parameters.heartbeat=180
            parameters.connection_attempts=1
            parameters.blocked_connection_timeout=15
            connection=pika.BlockingConnection(parameters)
            channel=connection.channel()
            declare(channel,settings.queue_name)
            channel.basic_qos(prefetch_count=1)

            def receive(ch,method,properties,body):
                try:
                    message = json.loads(body)
                    identity = message['job_id']
                    if not isinstance(identity,str) or len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):
                        raise ValueError('invalid task id')
                except (ValueError,TypeError,KeyError):
                    log.error('Invalid queue message rejected')
                    ch.basic_nack(method.delivery_tag,requeue=False)
                    return
                try:
                    process_job(store,identity,runner)
                except Exception:
                    ch.basic_nack(method.delivery_tag,requeue=False)
                    raise
                ch.basic_ack(method.delivery_tag)

            channel.basic_consume(queue=settings.queue_name,on_message_callback=receive,auto_ack=False)
            log.info('GPU worker ready; prefetch=1, parallel routes within a job')
            channel.start_consuming()
        except BaseException as exc:
            stop.set()
            store.worker(False,f'工作端已停止：{type(exc).__name__}: {exc}')
            log.exception('Worker stopped without automatic restart')
            raise
        finally:
            stop.set()
            if connection is not None and connection.is_open:
                connection.close()
            runner.close()
            store.engine.dispose()


if __name__=='__main__':
    main()
