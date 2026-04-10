#!/usr/bin/env python3
"""Fix ONNX schema.h explicit constructor compatibility issue."""
import pathlib

schema_path = pathlib.Path("/build/_deps/onnx-src/onnx/defs/schema.h")
text = schema_path.read_text()

# Fix: change implicit conversion to direct initialization
# From:  ONNX_UNUSED = \
#            OpSchema(#name, __FILE__, __LINE__)
# To:    ONNX_UNUSED(
#            OpSchema(#name, __FILE__, __LINE__))
text = text.replace(
    "ONNX_UNUSED = \\\n      OpSchema(#name, __FILE__, __LINE__)",
    "ONNX_UNUSED(\n      OpSchema(#name, __FILE__, __LINE__))",
)

schema_path.write_text(text)
print("Patched ONNX schema.h")
