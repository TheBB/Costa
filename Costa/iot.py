from datetime import datetime, timezone
import time
from typing import Dict

from azure.iot.device import IoTHubDeviceClient, MethodResponse, MethodRequest
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.protocol.models.cloud_to_device_method import CloudToDeviceMethod


class InternalServerError(Exception):
    pass

class UnknownMethodError(Exception):
    pass


class IotServer:

    client: IoTHubDeviceClient

    def __init__(self, connection_str: str):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_str)

    def serve(self):
        self.client.on_method_request_received = self.method_called
        self.client.connect()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.client.shutdown()

    def method_called(self, request: MethodRequest):
        func = f'on_{request.name}'
        if hasattr(self, func):
            try:
                payload = getattr(self, func)(request.payload)
                status = 200
            except Exception as e:
                payload = {'error': str(e)}
                status = 500
        else:
            payload = {'error': f"Unknown method '{request.name}'"}
            status = 404
        payload['time'] = datetime.now(timezone.utc).isoformat()
        response = MethodResponse.create_from_method_request(request, status, payload)
        self.client.send_method_response(response)

    def on_ping(self, payload: Dict) -> Dict:
        return {}


class IotClient:

    registry: IoTHubRegistryManager

    def __init__(self, connection_str: str):
        self.registry = IoTHubRegistryManager(connection_str)

    def invoke(self, device: str, method: str, payload: Dict) -> Dict:
        payload = {**payload, 'time': datetime.now(timezone.utc).isoformat()}
        method = CloudToDeviceMethod(method_name=method, payload=payload)
        response = self.registry.invoke_device_method(device, method)
        if response is None:
            raise Exception("Expected repsponse but got 'None'")
        if response.status == 500:
            raise InternalServerError(response.payload['error'])
        if response.status == 404:
            raise UnknownMethodError(response.payload['error'])
        if response.status != 200:
            raise Exception(f"Unexpected status code: {response.status}")
        return response.payload
