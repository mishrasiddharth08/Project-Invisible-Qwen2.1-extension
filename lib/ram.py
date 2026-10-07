"""Dependency-free system-RAM preflight; no GPU access or downloads."""
import os
import json
import struct
from pathlib import Path


def available_bytes():
    if os.name == 'nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
        state=Status();state.length=ctypes.sizeof(state)
        return int(state.available) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)) else None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
        return int(os.sysconf('SC_AVPHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))
    except (OSError,ValueError,AttributeError):return None


def packed_factor(path):
    name=str(path).lower()
    factor=4 if any(t in name for t in ('w4a8','nvfp4','w4a4')) else 2 if any(t in name for t in ('int8','convrot','fp8','mxfp8')) else 1
    try:
        with Path(path).open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            if not 2 <= size <= 64*2**20:return factor
            header=json.loads(f.read(size))
        metadata=str(header.get('__metadata__',{})).lower()
        if any(t in metadata for t in ('w4a8','nvfp4','w4a4')):return 4
        packed=any(k.endswith('.comfy_quant') or (k.endswith('.weight') and isinstance(v,dict) and v.get('dtype') in ('I8','U8')) for k,v in header.items())
        if any(k.endswith('.comfy_quant') for k in header) and factor==1:return 4
        if packed or '_quantization_metadata' in metadata:return max(2,factor)
    except (OSError,ValueError,struct.error):pass
    return factor


def preflight(bundle,prof=None,*,dequantize=False,available=None):
    prof=prof or {};paths=list((bundle.get('files') or {}).values())
    if not paths and bundle.get('folder'):
        paths=list(Path(bundle['folder']).rglob('*.safetensors'))
    sizes=[];temporary=0
    for path in {Path(p).resolve() for p in paths if p}:
        try:size=path.stat().st_size
        except OSError:continue
        factor=packed_factor(path)
        expand=bool(prof.get('portable') or dequantize) and factor>1
        sizes.append(size*factor if expand else size)
        if expand:temporary=max(temporary,size)
    if not sizes:return None
    weights=sum(sizes)
    overhead=2*2**30 if weights>=2**30 else 128*2**20
    required=weights+temporary+overhead
    free=available_bytes() if available is None else int(available)
    report={'required_bytes':required,'available_bytes':free}
    if free is not None and required>free:
        message=(f'Qwen model loading estimates a need for {required/2**30:.1f} GB of available system RAM; '
                 f'only {free/2**30:.1f} GB is free. Close other model workers/apps or use supported smaller weights. '
                 'GPU offload still needs system RAM. Expert config allow_ram_overcommit=true permits paging, which can be very slow.')
        if not prof.get('allow_ram_overcommit'):raise RuntimeError(message)
        print('[PI-Qwen21] RAM overcommit enabled: '+message)
    return report
