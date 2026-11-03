"""SQLAlchemy records shared by HTTP requests and the single GPU consumer."""
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, create_engine, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    kind: Mapped[str] = mapped_column(String(32), default='comparison')
    parent_id: Mapped[str | None] = mapped_column(ForeignKey('jobs.id'), nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(80), default='猫的照片')
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(24), default='queued', index=True)
    message: Mapped[str] = mapped_column(Text, default='等待模型处理')
    dispatched: Mapped[bool] = mapped_column(Boolean, default=False)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)


class WorkerState(Base):
    __tablename__ = 'worker_state'
    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    ready: Mapped[bool] = mapped_column(Boolean, default=False)
    message: Mapped[str] = mapped_column(Text)
    heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def serialize(job):
    return {'id':job.id,'kind':job.kind,'parent_id':job.parent_id,'label':job.label,'question':job.question,
            'state':job.state,'message':job.message,'result':job.result,
            'created_utc':job.created_at.isoformat(),'updated_utc':job.updated_at.isoformat(),
            'input_url':f'/api/files/{job.id}/input.png',
            'visualization_url':f'/api/files/{job.id}/traditional.png'}


class Store:
    def __init__(self, settings):
        self.settings = settings
        settings.prepare()
        options = {'check_same_thread':False, 'timeout':30} if settings.database_url.startswith('sqlite') else {}
        self.engine = create_engine(settings.database_url, connect_args=options, pool_pre_ping=True)
        self.session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        try:
            with self.session.begin() as db:
                if db.get(WorkerState,'gpu') is None:
                    db.add(WorkerState(name='gpu',ready=False,message='模型工作端尚未启动'))
        except IntegrityError:
            # A simultaneously starting process already created the singleton row.
            pass

    def add_user(self, username, password_hash):
        with self.session.begin() as db:
            user = User(id=uuid4().hex, username=username, password_hash=password_hash)
            db.add(user)
        return user

    def user(self, *, username=None, identity=None):
        with self.session() as db:
            return db.scalar(select(User).where(User.username == username)) if username is not None else db.get(User, identity)

    def add_job(self, user_id, *, label='猫的照片', kind='comparison', parent_id=None, question=None, identity=None):
        with self.session.begin() as db:
            if self.engine.dialect.name=='sqlite':
                db.execute(text('BEGIN IMMEDIATE'))
            else:
                db.scalar(select(WorkerState).where(WorkerState.name=='gpu').with_for_update())
            pending = db.scalar(select(func.count()).select_from(Job).where(Job.state.in_(['queued','running'])))
            if pending >= self.settings.max_pending:
                raise OverflowError('已有任务等待处理，请稍后再试')
            job = Job(id=identity or uuid4().hex, user_id=user_id, kind=kind, parent_id=parent_id,
                      question=question, label=label)
            db.add(job)
            db.flush()
            result = serialize(job)
        return result

    def get(self, identity, user_id=None):
        with self.session() as db:
            query = select(Job).where(Job.id==identity, Job.hidden.is_(False))
            if user_id is not None:
                query = query.where(Job.user_id==user_id)
            job = db.scalar(query)
            return serialize(job) if job else None

    def history(self, user_id, limit=30, offset=0, parent_id=None):
        with self.session() as db:
            query = select(Job).where(Job.user_id==user_id, Job.hidden.is_(False))
            query = query.where(Job.parent_id==parent_id) if parent_id else query.where(Job.kind=='comparison')
            total = db.scalar(select(func.count()).select_from(query.subquery()))
            items = db.scalars(query.order_by(Job.created_at.desc(),Job.id.desc()).limit(limit).offset(offset)).all()
            return {'items':[serialize(job) for job in items], 'total':total}

    def change(self, identity, **values):
        with self.session.begin() as db:
            db.execute(update(Job).where(Job.id==identity).values(**values, updated_at=now()))

    def hide(self, identity):
        with self.session.begin() as db:
            job = db.scalar(select(Job).where(Job.id==identity).with_for_update())
            children = db.scalars(select(Job).where(Job.parent_id==identity)).all()
            if any(item.state in {'queued','running'} for item in [job,*children]):
                raise ValueError('请等这张图片及问答任务结束后再移除记录')
            for item in [job,*children]:
                item.hidden = True

    def claim(self, identity):
        with self.session.begin() as db:
            job = db.scalar(select(Job).where(Job.id==identity, Job.hidden.is_(False)).with_for_update())
            if not job or job.state!='queued':
                return None
            job.state, job.message = 'running', '正在读取图片'
            db.flush()
            return serialize(job)

    def finish(self, identity, result, message):
        self.change(identity, state='completed', result=result, message=message)

    def interrupt_running(self):
        with self.session.begin() as db:
            db.execute(update(Job).where(Job.state=='running').values(state='interrupted',
                       message='工作进程中断；已保留记录，请查看日志后重新提交',updated_at=now()))

    def fail_unconfirmed(self):
        with self.session.begin() as db:
            db.execute(update(Job).where(Job.state=='queued',Job.dispatched.is_(False)).values(state='error',
                       message='上次任务投递未完成，请手动重新提交', updated_at=now()))

    def worker(self, ready=None, message=None):
        with self.session.begin() as db:
            record = db.get(WorkerState,'gpu')
            if ready is not None:
                if record is None:
                    record = WorkerState(name='gpu',ready=ready,message=message or '')
                    db.add(record)
                record.ready,record.message,record.heartbeat = ready,message or '',now()
                db.flush()
            if record is None:
                return {'ready':False,'message':'模型工作端尚未启动'}
            heartbeat = record.heartbeat.replace(tzinfo=timezone.utc) if record.heartbeat.tzinfo is None else record.heartbeat
            fresh = (now()-heartbeat).total_seconds()<30
            return {'ready':record.ready and fresh,'message':record.message if fresh else '模型工作端连接已中断',
                    'heartbeat':heartbeat.isoformat()}
