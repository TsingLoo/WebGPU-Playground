import json
import base64
import struct
import math
import os

latitudes = 32
longitudes = 32
vertices = []
normals = []
indices = []

for i in range(latitudes + 1):
    lat = i * math.pi / latitudes - math.pi / 2
    y = math.sin(lat)
    r = math.cos(lat)
    for j in range(longitudes + 1):
        lon = j * 2 * math.pi / longitudes
        x = r * math.cos(lon)
        z = r * math.sin(lon)
        vertices.extend([x, y, z])
        normals.extend([x, y, z])

for i in range(latitudes):
    for j in range(longitudes):
        first = i * (longitudes + 1) + j
        second = first + longitudes + 1
        indices.extend([first, second, first + 1])
        indices.extend([second, second + 1, first + 1])

vertex_bytes = struct.pack('<%df' % len(vertices), *vertices)
normal_bytes = struct.pack('<%df' % len(normals), *normals)
index_bytes = struct.pack('<%dI' % len(indices), *indices)

buffer_bytes = vertex_bytes + normal_bytes + index_bytes
buffer_b64 = base64.b64encode(buffer_bytes).decode('ascii')

gltf = {
    'asset': {'version': '2.0'},
    'scene': 0,
    'scenes': [{'nodes': [0]}],
    'nodes': [{'mesh': 0}],
    'meshes': [{'primitives': [{'attributes': {'POSITION': 1, 'NORMAL': 2}, 'indices': 0, 'material': 0}]}],
    'materials': [{
        'name': 'Glass',
        'pbrMetallicRoughness': {'baseColorFactor': [1,1,1,1], 'roughnessFactor': 0, 'metallicFactor': 0},
        'extensions': {
            'KHR_materials_transmission': {'transmissionFactor': 1},
            'KHR_materials_ior': {'ior': 1.5},
            'KHR_materials_volume': {'thicknessFactor': 2.0},
            'KHR_materials_dispersion': {'dispersion': 5.0}
        }
    }],
    'buffers': [{'uri': 'data:application/octet-stream;base64,' + buffer_b64, 'byteLength': len(buffer_bytes)}],
    'bufferViews': [
        {'buffer': 0, 'byteOffset': len(vertex_bytes) + len(normal_bytes), 'byteLength': len(index_bytes), 'target': 34963},
        {'buffer': 0, 'byteOffset': 0, 'byteLength': len(vertex_bytes), 'target': 34962},
        {'buffer': 0, 'byteOffset': len(vertex_bytes), 'byteLength': len(normal_bytes), 'target': 34962}
    ],
    'accessors': [
        {'bufferView': 0, 'byteOffset': 0, 'componentType': 5125, 'count': len(indices), 'type': 'SCALAR'},
        {'bufferView': 1, 'byteOffset': 0, 'componentType': 5126, 'count': len(vertices)//3, 'type': 'VEC3', 'min': [-1,-1,-1], 'max': [1,1,1]},
        {'bufferView': 2, 'byteOffset': 0, 'componentType': 5126, 'count': len(normals)//3, 'type': 'VEC3'}
    ],
    'extensionsUsed': ['KHR_materials_transmission', 'KHR_materials_ior', 'KHR_materials_volume', 'KHR_materials_dispersion']
}

os.makedirs('public/scenes/glass_sphere', exist_ok=True)
with open('public/scenes/glass_sphere/glass_sphere.gltf', 'w') as f:
    json.dump(gltf, f, indent=2)

with open('public/scenes/glass_sphere/scene.json', 'w') as f:
    json.dump({
      "name": "Glass Sphere",
      "models": [{ "path": "glass_sphere.gltf", "position": [0, 4, 0], "rotation": [0, 0, 0], "scale": [4, 4, 4] }],
      "lights": [{ "type": "directional", "direction": [1, -5, 1], "color": [1.0, 1.0, 1.0], "intensity": 4.0, "shadow": True }]
    }, f, indent=2)

print('Sphere saved.')
