from typing import Dict

from Costa.iot import IotServer


class MyServer(IotServer):

    def on_greet(self, payload: Dict) -> Dict:
        print('Received greeting to', payload['addressee'], 'at', payload['time'])
        return {'message': f"And hello to you too, {payload['addressee']}"}


cstr = "HostName=PoroTwinNor.azure-devices.net;DeviceId=PythonDevice;SharedAccessKey=x6WxHCGxx4EmG48UmvQjKaJGxPddem2OS53rfd+UwiQ="
server = MyServer(cstr)
server.serve()
