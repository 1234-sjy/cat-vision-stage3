"""Confirmed, persistent task delivery. No inference is retried here."""
import json
import pika


def declare(channel, name):
    channel.queue_declare(queue=name+'.failed', durable=True)
    channel.queue_declare(queue=name, durable=True, arguments={
        'x-dead-letter-exchange':'', 'x-dead-letter-routing-key':name+'.failed'})


class Publisher:
    def __init__(self, settings):
        self.settings = settings

    def publish(self, job_id):
        parameters = pika.URLParameters(self.settings.amqp_url)
        parameters.connection_attempts = 1
        parameters.socket_timeout = 5
        parameters.blocked_connection_timeout = 5
        with pika.BlockingConnection(parameters) as connection:
            channel = connection.channel()
            declare(channel,self.settings.queue_name)
            channel.confirm_delivery()
            channel.basic_publish(exchange='', routing_key=self.settings.queue_name, mandatory=True,
                body=json.dumps({'job_id':job_id}).encode(),
                properties=pika.BasicProperties(delivery_mode=2,content_type='application/json',message_id=job_id))
