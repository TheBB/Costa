from datetime import datetime, timezone
import json
from pathlib import Path
import time

from typing import Dict, Optional, Union

from azure.eventhub import EventHubConsumerClient, EventData
from azure.iot.device import IoTHubDeviceClient, MethodResponse, MethodRequest
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.protocol.models.cloud_to_device_method import CloudToDeviceMethod
from msrest.exceptions import HttpOperationError
import numpy as np

from .api import DataModel, DataTrainer, PhysicsModel


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '_object': 'array',
                '_shape': list(obj.shape),
                '_data': list(obj),
            }
        return super().default(self, obj)


def numpy_array_decoder(data: Dict):
    if '_object' not in data:
        return data
    if data['_object'] not in ('array',):
        return data
    return np.array(data['_data']).reshape(*data['_shape'])


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

    def __init__(self, connection_str: str):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_str)
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
        payload = json.loads(request.payload, object_hook=numpy_array_decoder)
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
        response = MethodResponse.create_from_method_request(
            request, status,
            json.dumps(payload_out, cls=NumpyArrayEncoder)
        )
        self.client.send_method_response(response)

    def emit(self, name: str, payload: Dict):
        """Emit a device-to-cloud message."""
        payload = {
            **payload,
            'name': name,
            'time': datetime.now(timezone.utc).isoformat()
        }
        self.client.send_message(json.dumps(payload, cls=NumpyArrayEncoder))

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

    def __init__(self, rstr: Optional[str] = None, hstr: Optional[str] = None):
        if rstr:
            self.registry = IoTHubRegistryManager(rstr)
        if hstr:
            self.hub = EventHubConsumerClient.from_connection_string(hstr, '$default')

    def invoke(self, device: str, method: str, payload: Dict) -> Dict:
        """Invoke a cloud-to-devic message and return its response.

        device: the device id of the addressee
        method: the name of the method to invoke
        payload: a parameter dictionary

        Returns the response dictionary, or raises UnknownMethodError or
        InternalServerError.
        """
        assert self.registry
        payload = {**payload, 'time': datetime.now(timezone.utc).isoformat()}
        payload = json.dumps(payload, cls=NumpyArrayEncoder)
        method = CloudToDeviceMethod(method_name=method, payload=payload)
        response = self.registry.invoke_device_method(device, method)
        if response is None:
            raise Exception("Expected repsponse but got 'None'")
        if response.status not in (200, 404, 500):
            raise Exception(f"Unexpected status code: {response.status}")
        payload = json.loads(response.payload, object_hook=numpy_array_decoder)
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
        assert self.hub
        self.hub.receive(on_event=self.on_event)

    def on_event(self, partition_context: int, event_data: EventData):
        """Callback for handling device-to-cloud messages.  This will invoke a
        method `on_name` where `name` is the message type, if one exists.  If it
        doesn't exist, the message is quietly ignored.
        """
        payload = json.loads(event_data.body_as_str(), object_hook=numpy_array_decoder)
        try:
            func_name = payload['name']
        except KeyError:
            return
        func = f'on_{func_name}'
        if hasattr(self, func):
            try:
                getattr(self, func)(payload)
            except Exception as e:
                print(e)


class PbmServer(IotServer):
    """A PbmServer exposes the functionality of a physics-based model to Azure IoT."""

    pbm: PhysicsModel

    def __init__(self, connection_str: str, pbm: PhysicsModel):
        self.pbm = pbm
        super().__init__(connection_str)

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

    def __init__(self, connection_str: str, device: str):
        self.device = device
        IotClient.__init__(self, rstr=connection_str)

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

    def __init__(self, connection_str: str, ddm: DataModel):
        self.ddm = ddm
        super().__init__(connection_str)

    def on_predict(self, payload: Dict) -> Dict:
        return {
            'sigma': self.ddm(payload['params'], payload['upred'])
        }


class DdmClient(DataModel, IotClient):
    """A DdmClient exposes the functionality of a remote data-driven model an Azure IoT
    as a regular DataModel object that can be used normally.
    """

    device: str

    def __init__(self, connection_str: str, device: str):
        self.device = device
        IotClient.__init__(self, connection_str)

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
        filename: Union[str, Path] = None,
        retrain_frequency: int = 5000,
        **kwargs
    ):
        super().__init__(hstr=hstr)
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
        state = payload['state']
        if self.prev_state is not None:
            self.trainer.append(payload['params'], self.prev_state, state)
            self.state_count += 1
            if self.state_count % self.retrain_frequency == 0:
                self.retrain()
        self.prev_state = state

    def on_clean_state(self, _):
        self.prev_state = None
