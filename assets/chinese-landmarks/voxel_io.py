"""Read and write the project's simple single-model VOX files."""
import struct
import numpy as np

def read_vox(path):
 data=path.read_bytes(); off=20
 while off+12<=len(data):
  tag,n,children=struct.unpack_from('<4sII',data,off); part=data[off+12:off+12+n]
  if tag==b'SIZE': size=struct.unpack('<III',part)
  elif tag==b'XYZI': vox=np.frombuffer(part[4:],np.uint8).reshape(-1,4).copy()
  elif tag==b'RGBA': pal=np.frombuffer(part,np.uint8).reshape(256,4).copy()
  off+=12+n+children
 return size,vox,pal

def native_vox(path,size,vox,pal):
 def chunk(tag,content): return tag+struct.pack('<II',len(content),0)+content
 children=chunk(b'SIZE',struct.pack('<III',*size))
 children+=chunk(b'XYZI',struct.pack('<I',len(vox))+vox.tobytes())
 children+=chunk(b'RGBA',np.concatenate([pal[1:],pal[:1]]).tobytes())
 path.write_bytes(b'VOX '+struct.pack('<I',150)+b'MAIN'+struct.pack('<II',0,len(children))+children)
