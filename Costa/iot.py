from datetime import datetime, timezone
import json
from pathlib import Path
import time

from typing import Dict, Optional, Union

from azure.eventhub import EventHubConsumerClient, EventData
from azure.iot.device import IoTHubDeviceClient, MethodResponse, MethodRequest
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.protocol.models.cloud_to_device_method import CloudToDeviceMethod
import numpy as np

from .api import DataModel, DataTrainer, PhysicsModel


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(self, obj)


class InternalServerError(Exception):
    pass

class UnknownMethodError(Exception):
    pass


class IotServer:

    client: IoTHubDeviceClient

    def __init__(self, connection_str: str):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_str)
        self.client.on_method_request_received = self.method_called

    def __enter__(self):
        self.client.connect()
        return self

    def __exit__(self, *args, **kwargs):
        self.client.shutdown()

    def wait(self):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    def method_called(self, request: MethodRequest):
        func = f'on_{request.name}'
        payload = json.loads(request.payload)
        if hasattr(self, func):
            try:
                payload = getattr(self, func)(payload)
                status = 200
            except Exception as e:
                payload = {'error': str(e)}
                status = 500
        else:
            payload = {'error': f"Unknown method '{request.name}'"}
            status = 404
        payload['time'] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(payload, cls=NumpyArrayEncoder)
        response = MethodResponse.create_from_method_request(request, status, payload)
        self.client.send_method_response(response)

    def emit(self, name: str, payload: Dict):
        payload = {
            **payload,
            'name': name,
            'time': datetime.now(timezone.utc).isoformat()
        }
        self.client.send_message(json.dumps(payload, cls=NumpyArrayEncoder))

    def on_ping(self, payload: Dict) -> Dict:
        return {}


class IotClient:

    registry: Optional[IoTHubRegistryManager] = None
    hub: Optional[EventHubConsumerClient] = None

    def __init__(self, rstr: Optional[str] = None, hstr: Optional[str] = None):
        if rstr:
            self.registry = IoTHubRegistryManager(rstr)
        if hstr:
            self.hub = EventHubConsumerClient.from_connection_string(hstr, '$default')

    def invoke(self, device: str, method: str, payload: Dict) -> Dict:
        assert self.registry
        payload = {**payload, 'time': datetime.now(timezone.utc).isoformat()}
        payload = json.dumps(payload, cls=NumpyArrayEncoder)
        method = CloudToDeviceMethod(method_name=method, payload=payload)
        response = self.registry.invoke_device_method(device, method)
        if response is None:
            raise Exception("Expected repsponse but got 'None'")
        if response.status not in (200, 404, 500):
            raise Exception(f"Unexpected status code: {response.status}")
        payload = json.loads(response.payload)
        if response.status == 500:
            raise InternalServerError(payload['error'])
        if response.status == 404:
            raise UnknownMethodError(payload['error'])
        return payload

    def listen(self):
        assert self.hub
        self.hub.receive(on_event=self.on_event)

    def on_event(self, partition_context: int, event_data: EventData):
        payload = event_data.body_as_json()
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

    pbm: PhysicsModel

    def __init__(self, connection_str: str, pbm: PhysicsModel):
        self.pbm = pbm
        super().__init__(connection_str)

    def on_ndof(self, _) -> Dict:
        return {'ndof': self.pbm.ndof}

    def on_dirichlet_dofs(self, _) -> Dict:
        return {'dofs': self.pbm.dirichlet_dofs()}

    def on_initial_condition(self, payload: Dict) -> Dict:
        # Todo: what to do?
        return {
            # 'u': self.pbm.initial_condition(payload['params'])
            'u': self.pbm.anasol(payload['params'])['primary']
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

    device: str

    def __init__(self, connection_str: str, device: str):
        self.device = device
        IotClient.__init__(self, rstr=connection_str)

    @property
    def ndof(self):
        return self.invoke(self.device, 'ndof', {})['ndof']

    def dirichlet_dofs(self):
        return self.invoke(self.device, 'dirichlet_dofs', {})['dofs']

    def initial_condition(self, params):
        return self.invoke(self.device, 'initial_condition', {'params': params})['u']

    def predict(self, params, uprev):
        return self.invoke(self.device, 'predict', {'params': params, 'uprev': uprev})['predicted']

    def residual(self, params, uprev, unext):
        return self.invoke(self.device, 'residual', {'params': params, 'uprev': uprev, 'unext': unext})['residual']

    def correct(self, params, uprev, sigma):
        return self.invoke(self.device, 'correct', {'params': params, 'uprev': uprev, 'sigma': sigma})['corrected']


class DdmServer(IotServer):

    ddm: DataModel

    def __init__(self, connection_str: str, ddm: DataModel):
        self.ddm = ddm
        super().__init__(connection_str)

    def on_predict(self, payload: Dict) -> Dict:
        return {
            'sigma': self.ddm(payload['params'], payload['upred'])
        }


class DdmClient(DataModel, IotClient):

    device: str

    def __init__(self, connection_str: str, device: str):
        self.device = device
        IotClient.__init__(self, connection_str)

    def __call__(self, params, upred):
        return self.invoke(self.device, 'predict', {'params': params, 'upred': upred})['sigma']


class PhysicalDevice(IotServer):

    def emit_state(self, params: Dict, state: np.ndarray):
        self.emit('new_state', {'params': params, 'state': state})

    def emit_clean(self):
        self.emit('clean_state', {})


class DdmTrainer(IotClient):

    trainer: DataTrainer
    ddm_server: Optional[DdmServer] = None
    connection_string: str

    prev_state: Optional[np.ndarray] = None

    retrain_frequency: int
    state_count: int

    train_kwargs: Dict
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
        state = np.array(payload['state'])
        if self.prev_state is not None:
            self.trainer.append(payload['params'], self.prev_state, state)
            self.state_count += 1
            if self.state_count % self.retrain_frequency == 0:
                self.retrain()
        self.prev_state = state

    def on_clean_state(self, _):
        self.prev_state = None
