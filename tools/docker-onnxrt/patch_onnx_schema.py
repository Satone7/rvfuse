#!/usr/bin/env python3
"""Fix ONNX schema.h explicit constructor compatibility issue."""
import pathlib
import re

schema_path = pathlib.Path("/build/_deps/onnx-src/onnx/defs/schema.h")

if not schema_path.exists():
    print(f"ERROR: {schema_path} not found!")
    exit(1)

text = schema_path.read_text()

# Fix: change implicit conversion to direct initialization.
# Original (ONNX v1.18.0):
#   static ...OpSchemaRegisterOnce(name) ONNX_UNUSED = \
#       OpSchema(...)
# Also handle previously broken patch:
#   static ...OpSchemaRegisterOnce(name) ONNX_UNUSED( \
#       OpSchema(...))
pattern = re.compile(
    r'(static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce)'
    r'\(op_schema_register_once##name##Counter\) ONNX_UNUSED[=(] \\\n'
    r'(      )OpSchema\(#name, __FILE__, __LINE__\)'
)

replacement = (
    r'[[maybe_unused]] \1(op_schema_register_once##name##Counter)(\n'
    r'\2ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(OpSchema(#name, __FILE__, __LINE__)))'
)

new_text, count = pattern.subn(replacement, text)
print(f"Patched {count} occurrence(s) in {schema_path}")
if count == 0:
    print("WARNING: No matches found. Checking for ONNX_UNUSED...")
    idx = text.find("ONNX_UNUSED")
    if idx >= 0:
        print(repr(text[idx-20:idx+120]))
    else:
        print("ONNX_UNUSED not found in file!")
else:
    schema_path.write_text(new_text)
    verify = schema_path.read_text()
    if "ONNX_UNUSED = \\" in verify or "ONNX_UNUSED( \\" in verify:
        print("ERROR: Patch verification failed!")
        exit(1)
    print("Patch verified successfully.")
