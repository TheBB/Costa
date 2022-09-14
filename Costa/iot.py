from abc import abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import uuid

from typing import ContextManager, Dict, Optional, Union, BinaryIO, List

from azure.core.exceptions import AzureError
from azure.eventhub import EventHubConsumerClient, EventData
from azure.iot.device import IoTHubDeviceClient, MethodResponse, MethodRequest
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.protocol.models.cloud_to_device_method import CloudToDeviceMethod
from azure.storage.blob import BlobClient
from msrest.exceptions import HttpOperationError
import numpy as np

from .api import DataModel, DataTrainer, PhysicsModel, VectorData
from .util import Logger


Arrays = Union[
    List,
    np.ndarray,
    Dict[
        str,
        Union[List, np.ndarray],
    ],
]


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(self, obj)


class InternalServerError(Exception):
    pass

class UnknownMethodError(Exception):
    pass


class IotConfig:

    # Azure connection strings
    registry: Optional[str] = None
    hub: Optional[str] = None
    storage: Optional[str] = None
    container: Optional[str] = None

    devices: Dict[str, str]

    def __init__(self):
        env = os.environ
        self.registry = env.get('COSTA_RSTR')
        self.hub = env.get('COSTA_HSTR')
        self.storage = env.get('COSTA_SSTR')
        self.container = env.get('COSTA_CONTAINER')

        self.devices = {}
        for key, value in env.items():
            if key.startswith('COSTA_') and key.endswith('_CSTR'):
                try:
                    _, device = next(k for k in value.split(';') if k.startswith('DeviceId')).split('=')
                    self.devices[device] = value
                except (StopIteration, TypeError, ValueError):
                    pass


class IotMailman:
    """A superclass that has access to the storage facilities of Azure IoT.
    Both servers and clients make use of this.

    For a subclass to upload, either the client attribute must be set
    (which is automatic for servers), or both the sstr and container arguments
    must be given.

    For a subclass to download, the sstr argument must be given.
    """

    storage_str: Optional[str] = None
    container_name: Optional[str] = None

    client: Optional[IoTHubDeviceClient] = None

    def __init__(self, config: IotConfig):
        self.storage_str = config.storage
        self.container_name = config.container

    def _upload_data_as_server(self, name: str, fmt: str, data: BinaryIO) -> Dict:
        assert self.client is not None
        storage = self.client.get_storage_info_for_blob(name)
        assert storage
        sas_url = 'https://{hostname}/{container}/{blob}{token}'.format(
            hostname=storage['hostName'],
            container=storage['containerName'],
            blob=storage['blobName'],
            token=storage['sasToken'],
        )
        blob_client = BlobClient.from_blob_url(sas_url)

        try:
            with BlobClient.from_blob_url(sas_url) as blob_client:
                blob_client.upload_blob(data, overwrite=True)
            self.client.notify_blob_upload_status(storage['correlationId'], True, 200, f'OK: {name}')

            return {
                'type': 'file',
                'format': fmt,
                'time': datetime.now(timezone.utc).isoformat(),
                'container': storage['containerName'],
                'blob': storage['blobName'],
            }

        except AzureError as err:
            print(f'Error received when uploading file: {err}', file=sys.stderr)
            self.client.notify_blob_upload_status(storage['correlationId'], False, err.status_code, str(err))

            return {
                'type': 'file',
                'format': 'error',
                'message': str(err),
            }

    def _upload_data_as_client(self, name: str, fmt: str, data: BinaryIO) -> Dict:
        assert self.storage_str is not None
        assert self.container_name is not None

        try:
            with BlobClient.from_connection_string(
                conn_str=self.storage_str,
                container_name=self.container_name,
                blob_name=name
            ) as blob_client:
                blob_client.upload_blob(data, overwrite=True)

        except AzureError as err:
            return {
                'type': 'file',
                'format': 'error',
                'message': str(err),
            }

        return {
            'type': 'file',
            'format': fmt,
            'time': datetime.now(timezone.utc).isoformat(),
            'container': self.container_name,
            'blob': name,
        }

    def upload_data(self, name: str, fmt: str, data: BinaryIO) -> Dict:
        name = f'{name}-{uuid.uuid1()}.{fmt}'
        if hasattr(self, 'name'):
            name = f'{self.name}-{name}'

        if self.client is not None:
            return self._upload_data_as_server(name, fmt, data)
        return self._upload_data_as_client(name, fmt, data)

    def upload_file(self, filename: str) -> Dict:
        path = Path(filename)
        with open(filename, 'rb') as f:
            return self.upload_data(path.stem, path.suffix[1:], f)

    def upload_single_ndarray(self, name: str, data: Union[List, np.ndarray]) -> Dict:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        with BytesIO() as b:
            np.save(b, data, allow_pickle=False)
            b.seek(0)
            return self.upload_data(name, 'npy', b)

    def upload_multiple_ndarrays(self, name: str, data: Dict[str, Union[List, np.ndarray]]) -> Dict:
        data = {
            k: v if isinstance(v, np.ndarray) else np.array(v)
            for k, v in data.items()
        }
        with BytesIO() as b:
            np.savez(b, **data)
            b.seek(0)
            return self.upload_data(name, 'npz', b)

    def upload_ndarrays(self, name: str, data: Arrays) -> Dict:
        if isinstance(data, Dict):
            return self.upload_multiple_ndarrays(name, data)
        return self.upload_single_ndarray(name, data)

    @contextmanager
    def download(self, filedata: Dict) -> ContextManager[BytesIO]:
        assert filedata['type'] == 'file'
        assert filedata['format'] != 'error'
        assert self.storage_str is not None
        with BytesIO() as b:
            with BlobClient.from_connection_string(
                conn_str=self.storage_str,
                container_name=filedata['container'],
                blob_name=filedata['blob']
            ) as client:
                client.download_blob().readinto(b)
            b.seek(0)
            yield b

    def download_file(self, filedata: Dict, filename: str):
        with open(filename, 'wb') as f:
            with self.download(filedata) as g:
                f.write(g.getvalue())

    def download_single_ndarray(self, filedata: Dict) -> np.ndarray:
        assert filedata['format'] == 'npy'
        with self.download(filedata) as data:
            return np.load(data)

    def download_multiple_ndarrays(self, filedata: Dict) -> Dict[str, np.ndarray]:
        assert filedata['format'] == 'npz'
        with self.download(filedata) as data:
            return dict(np.load(data))

    def download_ndarrays(self, filedata: Dict) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if filedata['format'] == 'npy':
            return self.download_single_ndarray(filedata)
        return self.download_multiple_ndarrays(filedata)


class IotServer(IotMailman, Logger):
    """An Azure IoT-powered device or server.

    This object is a context manager, and should be used as

        with IotServer(connection_str) as server:
            # connection is live here

    While live, the device will respond to cloud-to-device method requests
    automatically.  To customize a response to such a request, subclass this
    class and implement one or more methods named `on_methodname`, where
    `methodname` is the name of the method.  Each such method receives a payload
    dictionary as an argument and should return a dictionary as a response.
    """

    client: IoTHubDeviceClient

    name: str

    def __init__(self, name: str, config: IotConfig):
        IotMailman.__init__(self, config)
        Logger.__init__(self, name)
        self.name = name
        self.client = IoTHubDeviceClient.create_from_connection_string(config.devices[name])
        self.client.on_method_request_received = self.method_called

    def __enter__(self):
        """Establish a connection to the Azure IoT hub."""
        self.client.connect()
        return self

    def __exit__(self, *args, **kwargs):
        """Close the connection."""
        self.client.shutdown()

    def wait(self):
        """Utility method for entering an infinite loop. During this, the device
        will respond to cloud-to-device requests.
        """
        self.log('Online')
        try:
            while True:
                time.sleep(0.1)
                self.wait_poll()
        except KeyboardInterrupt:
            pass

    def wait_poll(self):
        """Override this method to perform actions while waiting."""
        pass

    def method_called(self, request: MethodRequest):
        """Primary callback for responding to a cloud-to-device request.

        This method delegates the work to a method called `on_methodname`, where
        `methodname` is the name of the method.  That method should accept a
        paylod dictionary as an argument, and return a payload dictionary which
        will constitute the response.

        Errors in the delegate method will be interpreted as status code 500
        (internal server error), while missing delegate methods will be
        interpreted as status code 404.  Otherwise, the response will have
        status code 200.
        """
        func = f'on_{request.name}'
        payload = json.loads(request.payload)
        if hasattr(self, func):
            self.log('Received query:', request.name)
            try:
                payload_out = getattr(self, func)(payload)
                status = 200
            except Exception as e:
                payload_out = {'error': str(e)}
                status = 500
                self.log('Failure during callback:', e)
        else:
            payload_out = {'error': f"Unknown method '{request.name}'"}
            status = 404
            self.log('Received unknow query:', request.name)
        payload_out = {**payload_out, 'time': datetime.now(timezone.utc).isoformat()}
        response = MethodResponse.create_from_method_request(
            request, status,
            json.dumps(payload_out, cls=NumpyArrayEncoder)
        )
        self.client.send_method_response(response)

    def emit(self, name: str, payload: Dict):
        """Emit a device-to-cloud message."""
        self.log('Emitting', name)
        payload = {
            **payload,
            'name': name,
            'time': datetime.now(timezone.utc).isoformat()
        }
        self.client.send_message(json.dumps(payload, cls=NumpyArrayEncoder))

    def on_ping(self, payload: Dict) -> Dict:
        """Standard response to a ping."""
        return {}


class IotClient(IotMailman, Logger):
    """An Azure IoT-powered client object.

    A client does two things:
    - invoke cloud-to-device requests and obtain responses
    - listen to device-to-cloud messages

    Unlike the server, which responds to cloud-to-device requests automatically,
    the client must explicitly invoke the listen() method to collect
    device-to-cloud messages.

    To customize a handler for device-to-cloud messages, implement a `on_name`
    method, where `name` is the name of the type of message you wish to listen
    to.  This method should accept a payload dictionary as a parameter.  Any
    return value will be ignored.

    The rstr constructor parameter is optional, but required for invoking
    cloud-to-device requests.  Likewise, the hstr parameter is optional, but
    required for listening to device-to-cloud messages.
    """

    # The registry manager is used when invoking cloud-to-device requests
    registry: Optional[IoTHubRegistryManager] = None

    # The event hub consumer is used when listening to device-to-cloud messages
    hub: Optional[EventHubConsumerClient] = None

    log_type = 'client'

    def __init__(self, name: str, config: IotConfig):
        IotMailman.__init__(self, config)
        Logger.__init__(self, name)
        if config.registry:
            self.registry = IoTHubRegistryManager(config.registry)
        if config.hub:
            self.hub = EventHubConsumerClient.from_connection_string(config.hub, '$default')

    def invoke(self, device: str, method: str, payload: Dict) -> Dict:
        """Invoke a cloud-to-devic message and return its response.

        device: the device id of the addressee
        method: the name of the method to invoke
        payload: a parameter dictionary

        Returns the response dictionary, or raises UnknownMethodError or
        InternalServerError.
        """
        assert self.registry
        self.log('Invoking', method, 'on', device)
        payload = {**payload, 'time': datetime.now(timezone.utc).isoformat()}
        payload = json.dumps(payload, cls=NumpyArrayEncoder)
        method = CloudToDeviceMethod(method_name=method, payload=payload)
        response = self.registry.invoke_device_method(device, method)
        if response is None:
            self.log('Received unexpected response: none')
            raise Exception("Expected repsponse but got 'None'")
        if response.status not in (200, 404, 500):
            self.log('Received unexpected status code:', response.status)
            raise Exception(f"Unexpected status code: {response.status}")
        payload = json.loads(response.payload)
        if response.status == 500:
            self.log('Received status code 500:', payload['error'])
            raise InternalServerError(payload['error'])
        if response.status == 404:
            self.log('Received status code 404:', payload['error'])
            raise UnknownMethodError(payload['error'])
        return payload

    def ping(self, device: str) -> bool:
        """Pings a device on the Azure IoT hub, and checks whether it is connected."""
        try:
            response = self.invoke(device, 'ping', {})
        except (InternalServerError, UnknownMethodError, HttpOperationError):
            return False
        return True

    def listen(self):
        """Listen perpetually to device-to-cloud messages."""
        assert self.hub
        self.log('Online, listening for events')
        self.hub.receive(on_event=self.on_event)

    def on_event(self, partition_context: int, event_data: EventData):
        """Callback for handling device-to-cloud messages.  This will invoke a
        method `on_name` where `name` is the message type, if one exists.  If it
        doesn't exist, the message is quietly ignored.
        """
        payload = event_data.body_as_json()
        try:
            func_name = payload['name']
        except KeyError:
            return
        func = f'on_{func_name}'
        if hasattr(self, func):
            try:
                self.log('Received event:', payload['name'])
                getattr(self, func)(payload)
            except Exception as e:
                print('Error:', e)


class PbmServer(IotServer):
    """A PbmServer exposes the functionality of a physics-based model to Azure IoT."""

    pbm: PhysicsModel

    log_type = 'pbm'

    def __init__(self, name: str, config: IotConfig, pbm: PhysicsModel):
        self.pbm = pbm
        super().__init__(name, config)

    def on_ndof(self, _) -> Dict:
        return {'ndof': self.pbm.ndof}

    def on_dirichlet_dofs(self, _) -> Dict:
        dofs = self.pbm.dirichlet_dofs()
        return {'dofs': self.upload_ndarrays('dofs', dofs)}

    def on_initial_condition(self, payload: Dict) -> Dict:
        initial = self.pbm.initial_condition(payload['params'])
        return {'initial': self.upload_ndarrays('initial', initial)}

    def on_predict(self, payload: Dict) -> Dict:
        mu = payload['params']
        uprev = self.download_ndarrays(payload['uprev'])
        upred = self.pbm.predict(payload['params'], uprev)
        return {'predicted': self.upload_ndarrays('pbm-upred', upred)}

    def on_residual(self, payload: Dict) -> Dict:
        uprev = self.download_ndarrays(payload['uprev'])
        unext = self.download_ndarrays(payload['unext'])
        sigma = self.pbm.residual(payload['params'], uprev, unext)
        return {'residual': self.upload_ndarrays('residual', sigma)}

    def on_correct(self, payload: Dict) -> Dict:
        uprev = self.download_ndarrays(payload['uprev'])
        sigma = self.download_ndarrays(payload['sigma'])
        ucorr = self.pbm.correct(payload['params'], uprev, sigma)
        return {'corrected': self.upload_ndarrays('ucorr', ucorr)}

    def on_qi(self, payload: Dict) -> Dict:
        u = self.download_ndarrays(payload['u'])
        qi = self.pbm.qi(payload['params'], u, payload['name'])
        return {'qi': qi}


class PbmClient(PhysicsModel, IotClient):
    """A PbmClient exposes the functionality of a remote physics-based model on Azure IoT
    as a regular PhysicsModel object that can be used normally.
    """

    device: str

    def __init__(self, config: IotConfig, device: str, name: Optional[str] = None):
        self.device = device
        IotClient.__init__(self, name or 'PbmClient', config)

    def ping_remote(self) -> bool:
        return self.ping(self.device)

    @property
    def ndof(self):
        return self.invoke(self.device, 'ndof', {})['ndof']

    def dirichlet_dofs(self):
        ref = self.invoke(self.device, 'dirichlet_dofs', {})['dofs']
        return self.download_ndarrays(ref)

    def initial_condition(self, params):
        ref = self.invoke(self.device, 'initial_condition', {'params': params})['initial']
        return self.download_ndarrays(ref)

    def predict(self, params, uprev):
        ref = self.invoke(self.device, 'predict', {
            'params': params,
            'uprev': self.upload_ndarrays('uprev', uprev),
        })['predicted']
        return self.download_ndarrays(ref)

    def residual(self, params, uprev, unext):
        ref = self.invoke(self.device, 'residual', {
            'params': params,
            'uprev': self.upload_ndarrays('uprev', uprev),
            'unext': self.upload_ndarrays('unext', unext),
        })['residual']
        return self.download_ndarrays(ref)

    def correct(self, params, uprev, sigma):
        ref = self.invoke(self.device, 'correct', {
            'params': params,
            'uprev': self.upload_ndarrays('uprev', uprev),
            'sigma': self.upload_ndarrays('sigma', sigma),
        })['corrected']
        return self.download_ndarrays(ref)

    def qi(self, params, u, name):
        return self.invoke(self.device, 'qi', {
            'params': params,
            'u': self.upload_ndarrays('u', u),
            'name': name,
        })['qi']


class DdmServer(IotServer):
    """A DdmServer exposes the functionality of a data-driven model to Azure IoT."""

    ddm: DataModel

    log_type = 'ddm'

    def __init__(self, name: str, config: IotConfig, ddm: DataModel):
        self.ddm = ddm
        super().__init__(name, config)

    def on_predict(self, payload: Dict) -> Dict:
        upred = self.download_ndarrays(payload['upred'])
        sigma = self.ddm(payload['params'], upred)
        return {'sigma': self.upload_ndarrays('sigma', sigma)}


class DdmClient(DataModel, IotClient):
    """A DdmClient exposes the functionality of a remote data-driven model an Azure IoT
    as a regular DataModel object that can be used normally.
    """

    device: str

    @classmethod
    def from_file(cls, filename):
        raise NotImplementedError

    def __init__(self, config: IotConfig, device: str, name: Optional[str] = None):
        self.device = device
        IotClient.__init__(self, name or 'DdmClient', config)

    def ping_remote(self) -> bool:
        return self.ping(self.device)

    def __call__(self, params: Dict, upred: np.ndarray) -> np.ndarray:
        ref = self.invoke(self.device, 'predict', {
            'params': params,
            'upred': self.upload_ndarrays('ddm-upred', upred),
        })['sigma']
        return self.download_ndarrays(ref)


class PhysicalDevice(IotServer):
    """A PhysicalDevice represents a device that regularly emits state information."""

    log_type = 'device'

    def emit_state(self, params: Dict, field: str, state: np.ndarray):
        """Notify the cloud about a new state."""
        self.emit('new_state', {
            'params': params,
            'field': field,
            'state': self.upload_ndarrays('state', state)
        })

    def emit_refreshed(self):
        """Notify the cloud that the state has been completely refreshed."""
        self.emit('state_refreshed', {})

    def emit_clean(self):
        """Notify the cloud that the setup has changed and that the standard
        time sequence of states is interrupted.  That is, the following state is
        not a time-step advanced from the previous state.
        """
        self.emit('clean_state', {})


class OptimizingController(IotClient):
    """A controller that optimizes some quantity."""

    state: VectorData
    control: Dict[str, float]
    target: str

    def __init__(self, name: str, config: IotConfig, target: str, control: Dict[str, float]):
        super().__init__(name, config)
        self.state = {}
        self.control = control
        self.target = target

    def on_new_state(self, payload):
        field = payload['field']
        state = self.download_ndarrays(payload['state'])
        self.state[field] = state

    def on_state_refreshed(self, _):
        if self.optimize():
            self.invoke(self.target, 'control', {'params': self.control})

    @abstractmethod
    def optimize(self) -> bool:
        pass


class DdmTrainer(IotClient):
    """A DdmTrainer is an advanced IoT client that listens to messages
    recording new physical states, passing them on to a DataTrainer.  At
    regular intervals, a new data-driven model is trained on the existing
    data and exposed to the IoT automatically.
    """

    # The object that does the training of the DDM
    trainer: DataTrainer

    # The current DDM running on the IoT. This is switched out whenever re-training occurrs.
    ddm_server: Optional[DdmServer] = None

    # Connection string used to expose the DDM to IoT.
    connection_string: str

    # The previous state emitted by the physical device.
    prev_state: Optional[np.ndarray] = None

    # How many new states are needed before we train a new model.
    retrain_frequency: int

    # Keep track of the number of new states seen so far.
    state_count: int

    # Additional arguments passed on to the training method.
    train_kwargs: Dict

    # Optional filename where the latest DDM will be saved.
    filename: Union[str, Path]

    def __init__(
        self,
        trainer: DataTrainer,
        hstr: str,
        cstr: str,
        filename: Union[str, Path] = None,
        retrain_frequency: int = 5000,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name or 'Trainer', hstr=hstr)
        self.trainer = trainer
        self.connection_string = cstr
        self.retrain_frequency = retrain_frequency
        self.state_count = 0
        self.train_kwargs = kwargs
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.ddm_server:
            self.ddm_server.__exit__()

    def retrain(self):
        ddm = self.trainer.train()
        if self.filename:
            ddm.save(self.filename)
        if self.ddm_server:
            self.ddm_server.__exit__()
        self.ddm_server = DdmServer(self.connection_string, ddm).__enter__()

    def on_new_state(self, payload: Dict):
        state = self.download_ndarrays(payload['state'])
        if self.prev_state is not None:
            self.trainer.append(payload['params'], self.prev_state, state)
            self.state_count += 1
            if self.state_count % self.retrain_frequency == 0:
                self.retrain()
        self.prev_state = state

    def on_clean_state(self, _):
        self.prev_state = None
