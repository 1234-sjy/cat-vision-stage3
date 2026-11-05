"""Authenticated API; CPU web processes never load inference weights."""
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import shutil
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError

from .common import BREEDS, load_image
from .stage3_auth import DUMMY_PASSWORD_HASH, create_token, hash_password, token_user, verify_password
from .stage3_queue import Publisher
from .stage3_settings import Settings
from .stage3_store import Store

log = logging.getLogger('catvision.api')


class Credentials(BaseModel):
    username: str = Field(min_length=3,max_length=32,pattern=r'^[\w-]+$')
    password: str = Field(min_length=8,max_length=128)

    @field_validator('username')
    @classmethod
    def normalize(cls, value):
        return value.casefold()


class Question(BaseModel):
    question: str = Field(min_length=1,max_length=500)

    @field_validator('question')
    @classmethod
    def clean(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('请输入图片问题')
        from .vlm_features import split_image_questions
        split_image_questions(value)
        return value


class Label(BaseModel):
    label: str = Field(min_length=1,max_length=80)

    @field_validator('label')
    @classmethod
    def clean(cls, value):
        if not value.strip():
            raise ValueError('名称不能为空')
        return value.strip()


def create_app(settings=None, publisher=None):
    settings = settings or Settings()
    store = Store(settings)
    publisher = publisher if publisher is not None else Publisher(settings)

    @asynccontextmanager
    async def lifespan(app):
        store.fail_unconfirmed()
        yield
        store.engine.dispose()

    app = FastAPI(title='看图识猫 · 第三阶段',version='0.3.0',lifespan=lifespan)
    app.state.store,app.state.publisher = store,publisher
    bearer = HTTPBearer(auto_error=False)

    def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not credentials or credentials.scheme.lower()!='bearer':
            raise HTTPException(401,'请先登录',headers={'WWW-Authenticate':'Bearer'})
        try:
            identity = token_user(credentials.credentials,settings)
            user = store.user(identity=identity)
        except jwt.PyJWTError:
            user = None
        if user is None:
            raise HTTPException(401,'登录已过期，请重新登录',headers={'WWW-Authenticate':'Bearer'})
        return user

    def owned(identity,user):
        if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):
            raise HTTPException(404,'没有找到这次记录')
        job = store.get(identity,user.id)
        if job is None:
            raise HTTPException(404,'没有找到这次记录')
        return job

    def token_response(user):
        return {'access_token':create_token(user.id,settings),'token_type':'bearer',
                'user':{'id':user.id,'username':user.username},'expires_in':settings.token_hours*3600}

    @app.post('/api/auth/register',status_code=201)
    def register(payload: Credentials):
        try:
            user = store.add_user(payload.username,hash_password(payload.password))
        except IntegrityError:
            raise HTTPException(409,'这个用户名已经注册')
        return token_response(user)

    @app.post('/api/auth/login')
    def login(payload: Credentials):
        user = store.user(username=payload.username)
        # Equal work also for an unknown username.
        expected = user.password_hash if user else DUMMY_PASSWORD_HASH
        matched = verify_password(payload.password,expected)
        if not user or not matched:
            raise HTTPException(401,'用户名或密码不正确')
        return token_response(user)

    @app.get('/api/auth/me')
    def me(user=Depends(current_user)):
        return {'id':user.id,'username':user.username}

    @app.get('/api/health')
    def health():
        return {'api':'ready','worker':store.worker()}

    @app.get('/api/info')
    def info(user=Depends(current_user)):
        return {'breeds':list(BREEDS.values()),'worker':store.worker(),'mode':'linux_docker_gpu'}

    def dispatch(job):
        try:
            publisher.publish(job['id'])
            store.change(job['id'],dispatched=True)
        except Exception:
            log.exception('Task publish failed: %s',job['id'])
            store.change(job['id'],state='error',message='任务投递失败，记录已保存；请稍后重新提交')
            raise HTTPException(503,'任务投递失败，记录已保存')
        return {'id':job['id'],'state':'queued'}

    def new_job(user,**values):
        try:
            return store.add_job(user.id,**values)
        except OverflowError as exc:
            raise HTTPException(429,str(exc)) from exc

    def accept_image(content,name,user):
        try:
            image = load_image(content,max_side=1600)
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        identity = uuid4().hex
        label = Path(name or '猫的照片').stem[:80] or '猫的照片'
        job = new_job(user,identity=identity,label=label)
        directory = settings.data_dir/identity
        try:
            directory.mkdir()
            image.save(directory/'input.png')
        except OSError:
            log.exception('Accepted image could not be saved: %s',identity)
            store.change(identity,state='error',message='原图保存失败，任务未投递；请查看服务器日志')
            raise HTTPException(503,'原图保存失败，记录已保存')
        return dispatch(job)

    @app.post('/api/recognize',status_code=202)
    async def recognize(file: UploadFile,user=Depends(current_user)):
        content = await file.read(12*1024*1024+1)
        if len(content)>12*1024*1024:
            raise HTTPException(413,'图片超过12MB，请缩小后重试')
        return await run_in_threadpool(accept_image,content,file.filename,user)

    @app.get('/api/jobs')
    def history(limit: int=Query(30,ge=1,le=100),offset: int=Query(0,ge=0),user=Depends(current_user)):
        return store.history(user.id,limit,offset)

    @app.get('/api/jobs/{identity}')
    def job(identity: str,user=Depends(current_user)):
        result = owned(identity,user)
        if result['state'] in {'queued','running'}:
            result['worker'] = store.worker()
        if result['kind']=='comparison':
            result['questions'] = store.history(user.id,100,parent_id=identity)['items']
        return result

    @app.patch('/api/jobs/{identity}')
    def rename(identity: str,payload: Label,user=Depends(current_user)):
        owned(identity,user)
        store.change(identity,label=payload.label)
        return owned(identity,user)

    @app.delete('/api/jobs/{identity}',status_code=204)
    def hide(identity: str,user=Depends(current_user)):
        owned(identity,user)
        try:
            store.hide(identity)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc
        return Response(status_code=204)

    @app.post('/api/jobs/{identity}/questions',status_code=202)
    def ask(identity: str,payload: Question,user=Depends(current_user)):
        parent = owned(identity,user)
        if parent['state']!='completed' or parent['kind']!='comparison':
            raise HTTPException(409,'请先完成这张图片的识别')
        if not (settings.data_dir/identity/'input.png').is_file():
            raise HTTPException(404,'没有找到这张原图')
        queued = new_job(user,kind='image_question',parent_id=identity,question=payload.question,label=parent['label'])
        return dispatch(queued)

    @app.post('/api/jobs/{identity}/retry',status_code=202)
    def retry(identity: str,user=Depends(current_user)):
        previous = owned(identity,user)
        if previous['state'] not in {'error','interrupted'}:
            raise HTTPException(409,'只能重新提交失败或中断的任务')
        if previous['kind']=='image_question':
            return ask(previous['parent_id'],Question(question=previous['question']),user)
        source = settings.data_dir/identity/'input.png'
        if not source.is_file():
            raise HTTPException(404,'没有找到这张原图')
        new_id = uuid4().hex
        queued = new_job(user,identity=new_id,label=previous['label'])
        target = settings.data_dir/new_id
        try:
            target.mkdir()
            shutil.copy2(source,target/'input.png')
        except OSError:
            log.exception('Retry image could not be saved: %s',new_id)
            store.change(new_id,state='error',message='原图复制失败，任务未投递；请查看服务器日志')
            raise HTTPException(503,'原图复制失败，记录已保存')
        return dispatch(queued)

    @app.get('/api/files/{identity}/{filename}')
    def file(identity: str,filename: str,user=Depends(current_user)):
        owned(identity,user)
        if filename not in {'input.png','traditional.png','subject.png'}:
            raise HTTPException(404,'没有找到图片')
        path = settings.data_dir/identity/filename
        if not path.is_file():
            raise HTTPException(404,'图片尚未生成')
        return FileResponse(path,media_type='image/png',headers={'Cache-Control':'private, no-store'})

    return app


if __name__=='__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(),host='0.0.0.0',port=8000)
