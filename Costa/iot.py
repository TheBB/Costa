from datetime import datetime, timezone

from functools import partial
from io import BytesIO
import json
from pathlib import Path
import time
import uuid

from typing import Dict, Optional, Union

from azure.core.exceptions import AzureError
from azure.eventhub import EventHubConsumerClient, EventData
from azure.iot.device import IoTHubDeviceClient, MethodResponse, MethodRequest
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.protocol.models.cloud_to_device_method import CloudToDeviceMethod
from azure.storage.blob import BlobClient
from msrest.exceptions import HttpOperationError
import numpy as np

from .api import DataModel, DataTrainer, PhysicsModel


class NumpyArrayEncoder(json.JSONEncoder):

    client: Optional[IoTHubDeviceClient]
    conn_str: Optional[str]
    container: Optional[str]
    size_threshold: int

    def __init__(
        self,
        client: Optional[IoTHubDeviceClient] = None,
        conn_str: Optional[str] = None,
        container: Optional[str] = None,
        size_threshold: int = 8000,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.client = client
        self.conn_str = conn_str
        self.container = container
        self.size_threshold = size_threshold

    def default(self, obj):
        if not isinstance(obj, np.ndarray):
            return super().default(obj)

        if obj.size <= self.size_threshold:
            return {
                '_object': 'array',
                '_shape': list(obj.shape),
                '_data': list(obj),
            }

        path = str(uuid.uuid1())
        if self.client is not None:
            container, blob = self.upload_via_client(obj.data, path)
        else:
            container, blob = self.upload_via_cstr(obj.data, path)

        z = {
            '_object': 'array-file',
            '_shape': list(obj.shape),
            '_container': container,
            '_blob': blob,
        }
        return z

    def upload_via_client(self, data, path):
        assert self.client, 'D'

        storage_data = self.client.get_storage_info_for_blob(path)
        assert storage_data is not None, 'E'

        container_name = storage_data['containerName']
        blob_name = storage_data['blobName']
        corr_id = storage_data['correlationId']

        sas_url = 'https://{hostname}/{container}/{blob}{token}'.format(
            hostname=storage_data['hostName'],
            container=container_name,
            blob=blob_name,
            token=storage_data['sasToken'],
        )
        blob_client = BlobClient.from_blob_url(sas_url)

        try:
            self.upload(data, blob_client)
            self.client.notify_blob_upload_status(corr_id, True, 200, f'OK: {path}')
        except AzureError as err:
            self.client.notify_blob_upload_status(corr_id, False, err.status_code, str(err))
            raise

        return container_name, blob_name

    def upload_via_cstr(self, data, path):
        assert self.conn_str, 'F'
        assert self.container, 'G'

        blob_client = BlobClient.from_connection_string(
            conn_str=self.conn_str,
            container_name=self.container,
            blob_name=path,
        )

        self.upload(data, blob_client)
        return self.container, path

    def upload(self, data, blob_client: BlobClient):
        with BytesIO(data) as src:
            with blob_client as tgt:
                tgt.upload_blob(src, overwrite=True)


def numpy_array_decoder(data: Dict, sstr: Optional[str] = None):
    if '_object' not in data:
        return data
    if data['_object'] == 'array':
        return np.array(data['_data']).reshape(*data['_shape'])
    if data['_object'] == 'array-file':
        assert sstr, 'A'
        with BlobClient.from_connection_string(sstr, data['_container'], data['_blob']) as blob_client:
            contents = blob_client.download_blob().readall()
        nelements = np.prod(data['_shape'])
        return np.frombuffer(contents, count=nelements).reshape(*data['_shape']).copy()


class InternalServerError(Exception):
    pass

class UnknownMethodError(Exception):
    pass


class IotServer:
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
    sstr: Optional[str]

    def __init__(self, connection_str: str, sstr: Optional[str] = None):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_str)
        self.client.on_method_request_received = self.method_called
        self.sstr = sstr

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
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
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
        decoder = partial(numpy_array_decoder, sstr=self.sstr)
        payload = json.loads(request.payload, object_hook=decoder)
        if hasattr(self, func):
            try:
                payload_out = getattr(self, func)(payload)
                status = 200
            except Exception as e:
                payload_out = {'error': str(e)}
                status = 500
        else:
            payload_out = {'error': f"Unknown method '{request.name}'"}
            status = 404
        payload_out = {**payload_out, 'time': datetime.now(timezone.utc).isoformat()}
        encoder = partial(NumpyArrayEncoder, client=self.client)
        response = MethodResponse.create_from_method_request(
            request, status,
            json.dumps(payload_out, cls=encoder)
        )
        self.client.send_method_response(response)

    def emit(self, name: str, payload: Dict):
        """Emit a device-to-cloud message."""
        payload = {
            **payload,
            'name': name,
            'time': datetime.now(timezone.utc).isoformat()
        }
        encoder = partial(NumpyArrayEncoder, client=self.client)
        self.client.send_message(json.dumps(payload, cls=encoder))

    def on_ping(self, payload: Dict) -> Dict:
        """Standard response to a ping."""
        return {}


class IotClient:
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

    # The connection string to the storage account, and its container
    sstr: Optional[str]
    container: Optional[str]

    def __init__(
        self,
        rstr: Optional[str] = None,
        hstr: Optional[str] = None,
        sstr: Optional[str] = None,
        container: Optional[str] = None
    ):
        if rstr:
            self.registry = IoTHubRegistryManager(rstr)
        if hstr:
            self.hub = EventHubConsumerClient.from_connection_string(hstr, '$default')
        self.sstr = sstr
        self.container = container

    def invoke(self, device: str, method: str, payload: Dict) -> Dict:
        """Invoke a cloud-to-devic message and return its response.

        device: the device id of the addressee
        method: the name of the method to invoke
        payload: a parameter dictionary

        Returns the response dictionary, or raises UnknownMethodError or
        InternalServerError.
        """
        assert self.registry, 'B'
        payload = {**payload, 'time': datetime.now(timezone.utc).isoformat()}
        encoder = partial(NumpyArrayEncoder, conn_str=self.sstr, container=self.container)
        payload = json.dumps(payload, cls=encoder)
        method = CloudToDeviceMethod(method_name=method, payload=payload)
        response = self.registry.invoke_device_method(device, method)
        if response is None:
            raise Exception("Expected repsponse but got 'None'")
        if response.status not in (200, 404, 500):
            raise Exception(f"Unexpected status code: {response.status}")
        decoder = partial(numpy_array_decoder, sstr=self.sstr)
        payload = json.loads(response.payload, object_hook=decoder)
        if response.status == 500:
            raise InternalServerError(payload['error'])
        if response.status == 404:
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
        assert self.hub, 'C'
        self.hub.receive(on_event=self.on_event)

    def on_event(self, partition_context: int, event_data: EventData):
        """Callback for handling device-to-cloud messages.  This will invoke a
        method `on_name` where `name` is the message type, if one exists.  If it
        doesn't exist, the message is quietly ignored.
        """
        decoder = partial(numpy_array_decoder, sstr=self.sstr)
        payload = json.loads(event_data.body_as_str(), object_hook=decoder)
        try:
            func_name = payload['name']
        except KeyError:
            return
        func = f'on_{func_name}'
        if hasattr(self, func):
            try:
                getattr(self, func)(payload)
            except Exception as e:
                print(repr(e))


class PbmServer(IotServer):
    """A PbmServer exposes the functionality of a physics-based model to Azure IoT."""

    pbm: PhysicsModel

    def __init__(self, connection_str: str, pbm: PhysicsModel, sstr: Optional[str] = None):
        self.pbm = pbm
        super().__init__(connection_str, sstr=sstr)

    def on_ndof(self, _) -> Dict:
        return {'ndof': self.pbm.ndof}

    def on_dirichlet_dofs(self, _) -> Dict:
        return {'dofs': self.pbm.dirichlet_dofs()}

    def on_initial_condition(self, payload: Dict) -> Dict:
        return {
            'initial': self.pbm.initial_condition(payload['params'])
        }

    def on_predict(self, payload: Dict) -> Dict:
        return {
            'predicted': self.pbm.predict(payload['params'], payload['uprev'])
        }

    def on_residual(self, payload: Dict) -> Dict:
        return {
            'residual': self.pbm.residual(payload['params'], payload['uprev'], payload['unext'])
        }

    def on_correct(self, payload: Dict) -> Dict:
        return {
            'corrected': self.pbm.correct(payload['params'], payload['uprev'], payload['sigma'])
        }


class PbmClient(PhysicsModel, IotClient):
    """A PbmClient exposes the functionality of a remote physics-based model on Azure IoT
    as a regular PhysicsModel object that can be used normally.
    """

    device: str

    def __init__(self, connection_str: str, device: str, sstr: Optional[str] = None, container: Optional[str] = None):
        self.device = device
        IotClient.__init__(self, rstr=connection_str, sstr=sstr, container=container)

    def ping_remote(self) -> bool:
        return self.ping(self.device)

    @property
    def ndof(self):
        return self.invoke(self.device, 'ndof', {})['ndof']

    def dirichlet_dofs(self):
        return self.invoke(self.device, 'dirichlet_dofs', {})['dofs']

    def initial_condition(self, params):
        return self.invoke(self.device, 'initial_condition', {'params': params})['initial']

    def predict(self, params, uprev):
        return self.invoke(self.device, 'predict', {'params': params, 'uprev': uprev})['predicted']

    def residual(self, params, uprev, unext):
        return self.invoke(self.device, 'residual', {'params': params, 'uprev': uprev, 'unext': unext})['residual']

    def correct(self, params, uprev, sigma):
        return self.invoke(self.device, 'correct', {'params': params, 'uprev': uprev, 'sigma': sigma})['corrected']


class DdmServer(IotServer):
    """A DdmServer exposes the functionality of a data-driven model to Azure IoT."""

    ddm: DataModel

    def __init__(self, connection_str: str, ddm: DataModel, sstr: Optional[str] = None):
        self.ddm = ddm
        super().__init__(connection_str, sstr=sstr)

    def on_predict(self, payload: Dict) -> Dict:
        return {
            'sigma': self.ddm(payload['params'], payload['upred'])
        }


class DdmClient(DataModel, IotClient):
    """A DdmClient exposes the functionality of a remote data-driven model an Azure IoT
    as a regular DataModel object that can be used normally.
    """

    device: str

    def __init__(self, connection_str: str, device: str, sstr: Optional[str] = None, container: Optional[str] = None):
        self.device = device
        IotClient.__init__(self, rstr=connection_str, sstr=sstr, container=container)

    def ping_remote(self) -> bool:
        return self.ping(self.device)

    def __call__(self, params, upred):
        return self.invoke(self.device, 'predict', {'params': params, 'upred': upred})['sigma']


class PhysicalDevice(IotServer):
    """A PhysicalDevice represents a device that regularly emits state information."""

    def emit_state(self, params: Dict, state: np.ndarray):
        """Notify the cloud about a new state."""
        self.emit('new_state', {'params': params, 'state': state})

    def emit_clean(self):
        """Notify the cloud that the setup has changed and that the standard
        time sequence of states is interrupted.  That is, the following state is
        not a time-step advanced from the previous state.
        """
        self.emit('clean_state', {})


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
        sstr: str,
        container: str,
        filename: Union[str, Path] = None,
        retrain_frequency: int = 5000,
        **kwargs
    ):
        super().__init__(hstr=hstr, sstr=sstr, container=container)
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
        self.ddm_server = DdmServer(self.connection_string, ddm, sstr=self.sstr).__enter__()

    def on_new_state(self, payload: Dict):
        state = payload['state']
        if self.prev_state is not None:
            self.trainer.append(payload['params'], self.prev_state, state)
            self.state_count += 1
            if self.state_count % self.retrain_frequency == 0:
                self.retrain()
        self.prev_state = state

    def on_clean_state(self, _):
        self.prev_state = None
