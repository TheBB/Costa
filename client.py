from Costa.iot import IotClient


cstr = "HostName=PoroTwinNor.azure-devices.net;SharedAccessKeyName=service;SharedAccessKey=SZYJLbgSXpL6lVerwUSUXs///Z9gI/m/Gz3UCgfint8="
client = IotClient(cstr)
response = client.invoke('PythonDevice', 'greet', {'addressee': 1})
print('Received response:', response['message'], 'at', response['time'])
