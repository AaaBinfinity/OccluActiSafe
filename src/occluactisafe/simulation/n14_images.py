"""实际CARLA BGRA帧无损写PNG；Python路径支持中文，不依赖原生ANSI保存接口。"""
import hashlib
from pathlib import Path
import struct
import zlib


def save_rgb_frame(image, path):
    import numpy as np
    width,height=image.width,image.height;data=image.raw_data
    if width<=0 or height<=0 or len(data)!=width*height*4:raise ValueError('实际相机BGRA尺寸不符')
    raw=np.frombuffer(data,dtype=np.uint8).reshape(height,width,4)
    rgb=raw[:,:,:3][:,:,::-1].copy()
    scanlines=np.empty((height,1+width*3),dtype=np.uint8);scanlines[:,0]=0
    scanlines[:,1:]=rgb.reshape(height,width*3)
    def chunk(tag,value):return struct.pack('!I',len(value))+tag+value+struct.pack('!I',zlib.crc32(tag+value)&0xffffffff)
    encoded=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))
             +chunk(b'IDAT',zlib.compress(scanlines.tobytes(),3))+chunk(b'IEND',b''))
    path=Path(path);temporary=path.with_name('待保存_'+path.name)
    with temporary.open('xb') as stream:stream.write(encoded)
    temporary.replace(path)
    return {'native_size':[width,height],'raw_bgra_sha256':hashlib.sha256(data).hexdigest(),
            'rgb_pixels_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'sha256':hashlib.sha256(encoded).hexdigest()}
