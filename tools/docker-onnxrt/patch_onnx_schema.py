#!/usr/bin/env python3
"""Fix ONNX schema.h explicit constructor compatibility issue."""
import pathlib
import re

schema_path = pathlib.Path("/build/_deps/onnx-src/onnx/defs/schema.h")
text = schema_path.read_text()

# Fix: change implicit conversion to direct initialization.
# Original:
#   static ...::OpSchemaRegisterOnce(name) ONNX_UNUSED = \
#       OpSchema(...)
# Fixed:
#   [[maybe_unused]] static ...::OpSchemaRegisterOnce(name)( \
#       ...::OpSchemaRegisterOnce(OpSchema(...)))
#
# The OpSchemaRegisterOnce constructor is explicit, so we must call it directly.

# Match and replace the two-line macro body
pattern = re.compile(
    r'(static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce)'
    r'\(op_schema_register_once##name##Counter\) ONNX_UNUSED = \\\n'
    r'(      )OpSchema\(#name, __FILE__, __LINE__\)'
)

replacement = (
    r'[[maybe_unused]] \1(op_schema_register_once##name##Counter)(\n'
    r'\2ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(OpSchema(#name, __FILE__, __LINE__)))'
)

text = pattern.sub(replacement, text)
schema_path.write_text(text)
print("Patched ONNX schema.h")
