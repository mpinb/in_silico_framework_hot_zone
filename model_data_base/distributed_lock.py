import distributed
#import fasteners
import redis
import yaml
import os
import warnings

config_path = os.path.join(os.path.dirname(__file__), 'distributed_lock_settings.yaml')

if os.path.exists(config_path):
    config = yaml.load(config_path)
else:
    #config = [dict(type = 'redis', config = dict(host = 'spock', port = 8885, socket_timeout = 1)),
    #          dict(type = 'redis', config = dict(host = 'localhost', port = 6379, socket_timeout = 1))]
    config = [dict(type = 'file')]    


def get_client():
    for server in config:
        print 'trying to connect to distributed locking server {}'.format(str(server))
        if server['type'] == 'redis':
            import redis
            try:
                c = redis.StrictRedis(**server['config'])
                c.client_list()
                print ('success!')
                return server, c
            except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError):
                pass
        elif server['type'] == 'file':
            warnings.warn('Using file based locking.'
            'Please make sure that you only provide names, that can be filenames, ideally absolute paths.'
            'Please be careful on nfs mounts as file based locking has issues in this case.')
            return server, None
        else:
            raise NotImplementedError()
    raise RuntimeError('could not connect to a locking server.')

server, client = get_client()

def update_config(c):
    global server
    global client
    global config
    config = c
    server, client = get_client()

def get_lock(name):
    if server['type'] is 'file':
        import fasteners
        return fasteners.InterProcessLock(name)
    elif server['type'] == 'redis':
        return redis.lock.Lock(client, name, timeout = 300)