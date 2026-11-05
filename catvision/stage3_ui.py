"""Streamlit client for the authenticated API. This process has no GPU models."""
import os
import re
import requests
import streamlit as st

API = os.getenv('CATVISION_API_URL','http://127.0.0.1:8000').rstrip('/')
st.set_page_config(page_title='看图识猫',page_icon='🐈',layout='wide')


def request(method,path,**kwargs):
    token = st.session_state.get('token')
    headers = {'Authorization':'Bearer '+token} if token else {}
    response = requests.request(method,API+path,headers=headers,timeout=(5,45),**kwargs)
    if response.status_code==401 and token:
        st.session_state.clear()
        st.rerun()
    if not response.ok:
        try:
            detail = response.json().get('detail','请求未完成')
            if isinstance(detail,list):
                detail = '；'.join(item.get('msg','参数不正确') for item in detail)
        except ValueError:
            detail = '服务暂时不可用，请稍后再试'
        raise ValueError(detail)
    return response


def image(identity,filename):
    cache = st.session_state.setdefault('images',{})
    key = (identity,filename)
    if key not in cache:
        cache[key] = request('GET',f'/api/files/{identity}/{filename}').content
    return cache[key]


def upload_changed():
    if st.session_state.get('task_busy'):
        # Discard a late upload event from the previous widget instance.
        st.session_state.upload_epoch=st.session_state.get('upload_epoch',0)+1
        return
    st.session_state.active_job=None
    st.session_state.current_job=None
    st.session_state.pop('active_filename',None)


def activate(identity,filename=None):
    if not isinstance(identity,str) or not re.fullmatch('[0-9a-f]{32}',identity):
        raise ValueError('返回任务与当前图片不匹配，请刷新后重新提交。')
    st.session_state.active_job=identity
    st.session_state.current_job=None
    st.session_state.upload_epoch=st.session_state.get('upload_epoch',0)+1
    st.session_state.active_filename=filename


def validate_job(job,identity):
    if job.get('id')!=identity or job.get('kind')!='comparison':
        raise ValueError('返回任务与当前图片不匹配，请刷新后重新提交。')
    if job.get('state')=='completed':
        result=job.get('result') or {}
        if result.get('task_id')!=identity or not re.fullmatch('[0-9a-f]{64}',result.get('input_sha256','')):
            raise ValueError('结果任务与当前图片不匹配，请刷新后重新提交。')
    for question in job.get('questions',[]):
        if question.get('parent_id')!=identity or question.get('kind')!='image_question':
            raise ValueError('问答任务与当前图片不匹配。')
        if question.get('state')=='completed':
            answer=question.get('result') or {}
            if (answer.get('task_id')!=question.get('id') or answer.get('parent_id')!=identity or
                    answer.get('input_sha256')!=(job.get('result') or {}).get('input_sha256')):
                raise ValueError('问答结果任务与当前图片不匹配。')


def auth():
    st.title('看图识猫')
    st.write('用同一张照片，对照传统视觉与视觉大模型的结果。')
    left,middle,right = st.columns([1,2,1])
    with middle:
        login,register = st.tabs(['登录','注册'])
        for tab,action,label in [(login,'login','登录'),(register,'register','注册并登录')]:
            with tab,st.form('auth-'+action):
                username = st.text_input('用户名',key=action+'-username',max_chars=32)
                password = st.text_input('密码',type='password',key=action+'-password',max_chars=128)
                if action=='register':
                    st.caption('用户名3–32个字符，可用中文、字母、数字、下划线或短横线；密码至少8个字符。')
                submit = st.form_submit_button(label,type='primary',width='stretch')
                if submit:
                    try:
                        data = request('POST','/api/auth/'+action,json={'username':username,'password':password}).json()
                        st.session_state.token=data['access_token']
                        st.session_state.username=data['user']['username']
                        st.rerun()
                    except (ValueError,requests.RequestException) as exc:
                        st.error(str(exc))


def route_result(route,result,identity):
    labels = {'no_cat':'未检测到猫','unsupported':'其他品种','parse_error':'结果解析未完成','error':'本次识别未完成'}
    title = result.get('breed_zh') if result.get('status')=='ok' else labels.get(result.get('status'),'尚无结果')
    st.subheader(title or '尚无结果')
    if route=='traditional':
        try:
            st.image(image(identity,'traditional.png'),width='stretch')
        except ValueError:
            st.info('这次运行没有生成标注图。')
        score = result.get('breed_score')
        if isinstance(score,(int,float)):
            st.caption(f'这张图片的分类分数：{score:.1%}')
    else:
        for feature in result.get('features',[]):
            st.write('· '+feature)
        for key,label in [('pose','姿态与位置'),('scene','画面背景')]:
            if result.get(key):
                st.markdown('**'+label+'**')
                st.write(result[key])
        if result.get('observation_status') not in (None,'ok') and result.get('observation_raw_text'):
            st.write(result['observation_raw_text'])
    if result.get('error'):
        st.error(result['error'])
    if isinstance(result.get('inference_seconds'),(float,int)):
        st.caption(f"本次用时 {result['inference_seconds']:.2f} 秒")


@st.fragment(run_every=0.7)
def current_result():
    identity = st.session_state.get('active_job')
    if not identity:
        st.info('选择一张照片，点击“开始识别”。')
        return
    try:
        job = request('GET','/api/jobs/'+identity).json()
        if st.session_state.get('active_job')!=identity:
            return
        validate_job(job,identity)
        st.session_state.current_job=job
        pending = job['state'] in {'queued','running'} or any(
            item['state'] in {'queued','running'} for item in job.get('questions',[]))
        if st.session_state.get('task_busy',False)!=pending:
            st.session_state.task_busy=pending
            st.rerun()
        if job['state'] in {'queued','running'}:
            st.info(job['message'])
            if not job.get('worker',{}).get('ready',True):
                st.warning(job['worker']['message'])
            return
        if job['state'] in {'error','interrupted'}:
            st.error(job['message'])
            if st.button('重新提交这张图片',key='retry-'+identity):
                new = request('POST','/api/jobs/'+identity+'/retry').json()
                activate(new['id'],st.session_state.get('active_filename'))
                st.session_state.task_busy=True
                st.rerun()
            return
        result = job['result'] or {}
        left,right = st.columns(2,gap='large')
        with left,st.container(border=True):
            st.markdown('### A · 传统视觉')
            st.caption('定位、分割与品种分类')
            route_result('traditional',result.get('traditional',{}),identity)
        with right,st.container(border=True):
            st.markdown('### B · 视觉大模型')
            st.caption('读取原图，判断品种并观察画面')
            route_result('vlm',result.get('vlm',{}),identity)
        seconds = result.get('total_seconds')
        if isinstance(seconds,(float,int)):
            st.success(f'识别完成 · 两条路线并行用时 {seconds:.2f} 秒')
        st.markdown('#### 再问这张图')
        questions = list(reversed(job.get('questions',[])))
        pending = any(q['state'] in {'queued','running'} for q in questions)
        for item in questions:
            with st.container(border=True):
                st.markdown('**问：**'+item['question'])
                if item['state']=='completed':
                    st.write((item['result'] or {}).get('answer','本次没有返回回答'))
                    value = (item['result'] or {}).get('inference_seconds')
                    if isinstance(value,(float,int)):
                        st.caption(f'问答用时 {value:.2f} 秒')
                elif item['state'] in {'queued','running'}:
                    st.info(item['message'])
                else:
                    st.error(item['message'])
        with st.form('question-'+identity,clear_on_submit=True):
            question = st.text_input('关于这张图片的问题',max_chars=500,placeholder='例如：这只猫的耳朵有什么特点？')
            if st.form_submit_button('提问',disabled=pending):
                if question.strip():
                    st.session_state.task_busy=True
                    try:
                        request('POST','/api/jobs/'+identity+'/questions',json={'question':question})
                    except (ValueError,requests.RequestException):
                        st.session_state.task_busy=False
                        raise
                    st.rerun()
        with st.expander('整理这条记录'):
            label = st.text_input('照片名称',value=job['label'],key='label-'+identity,max_chars=80)
            if st.button('保存名称',key='rename-'+identity):
                request('PATCH','/api/jobs/'+identity,json={'label':label})
                st.success('名称已保存')
            confirmed = st.checkbox('确认从我的历史列表移除',key='hide-confirm-'+identity)
            if st.button('从列表移除',disabled=not confirmed or pending,key='hide-'+identity):
                request('DELETE','/api/jobs/'+identity)
                st.session_state.active_job=None
                st.session_state.current_job=None
                st.session_state.pop('active_filename',None)
                st.rerun()
    except (ValueError,requests.RequestException) as exc:
        st.error(str(exc))


def main():
    if not st.session_state.get('token'):
        auth()
        return
    with st.sidebar:
        st.title('看图识猫')
        st.caption('当前用户：'+st.session_state.get('username',''))
        if st.button('退出登录'):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.subheader('我的历史照片')
        if st.button('刷新历史'):
            st.rerun()
        try:
            history = request('GET','/api/jobs?limit=30').json()
            if not history['items']:
                st.caption('还没有识别记录。')
            for job in history['items']:
                caption = {'completed':'完成','error':'失败','interrupted':'中断','queued':'等待','running':'处理中'}.get(job['state'],job['state'])
                if st.button(job['label']+' · '+caption,key='open-'+job['id'],width='stretch',
                             disabled=st.session_state.get('task_busy',False)):
                    activate(job['id'])
                    st.rerun()
        except (ValueError,requests.RequestException) as exc:
            st.error(str(exc))
    st.title('一张猫图，两个答案。')
    st.write('对照品种识别与原图观察，还能继续提问。')
    try:
        info = request('GET','/api/info').json()
    except (ValueError,requests.RequestException) as exc:
        st.error(str(exc))
        return
    if not info['worker']['ready']:
        st.info(info['worker']['message'])
        if st.button('刷新模型状态'):
            st.rerun()
    photo_column,result_column = st.columns([1,2.5],gap='large')
    with photo_column:
        st.subheader('共同输入')
        busy=st.session_state.get('task_busy',False)
        epoch=st.session_state.get('upload_epoch',0)
        uploaded = st.file_uploader('选择猫的照片',type=['jpg','jpeg','png','webp'],
                                    key=f'upload-{epoch}',disabled=busy,on_change=upload_changed)
        if st.session_state.get('active_job'):
            try:
                st.image(image(st.session_state.active_job,'input.png'),width='stretch')
                current=st.session_state.get('current_job') or {}
                caption=st.session_state.get('active_filename') or current.get('label')
                if caption:
                    st.caption(caption)
            except (ValueError,requests.RequestException):
                pass
        elif uploaded:
            st.image(uploaded,width='stretch')
            if st.button('开始识别',type='primary',width='stretch',disabled=busy or not info['worker']['ready']):
                st.session_state.task_busy=True
                try:
                    job = request('POST','/api/recognize',files={'file':(uploaded.name,uploaded.getvalue(),uploaded.type)}).json()
                    activate(job['id'],uploaded.name)
                    st.rerun()
                except (ValueError,requests.RequestException) as exc:
                    st.session_state.task_busy=False
                    st.error(str(exc))
        st.caption('支持JPG、PNG、WebP，单张不超过12MB。')
    with result_column:
        current_result()
    with st.expander('方法与支持品种'):
        st.write('传统视觉使用YOLO检测、SAM分割和Swin分类器；视觉大模型独立读取同一原图，输出一个品种，再描述外观、姿态和背景。两条路线不传递答案。')
        st.write('支持品种：'+'、'.join(info['breeds'])+'。')


if __name__=='__main__':
    main()
