"""Backend-facing detector adapters.

Heavy model runtimes remain under inference/ because they carry runtime-local
imports, setup tools, and portability notes. Backend code should call the
adapters here rather than constructing runtime commands directly.
"""
