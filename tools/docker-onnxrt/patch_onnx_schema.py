#!/usr/bin/env python3
"""Fix ONNX schema.h explicit constructor compatibility issue."""
import pathlib

schema_path = pathlib.Path("/build/_deps/onnx-src/onnx/defs/schema.h")

if not schema_path.exists():
    print(f"ERROR: {schema_path} not found!")
    exit(1)

text = schema_path.read_text()

# Build search strings using chr() to avoid source code escaping issues.
# chr(92) = backslash, chr(10) = newline
bs = chr(92)
nl = chr(10)

replacements = [
    (
        # Original ONNX v1.18.0: "ONNX_UNUSED = \" + newline + "      OpSchema(...)"
        "ONNX_UNUSED = " + bs + nl + "      OpSchema(#name, __FILE__, __LINE__)",
        "[[maybe_unused]] static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(op_schema_register_once##name##Counter)(" + nl +
        "      ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(OpSchema(#name, __FILE__, __LINE__)))",
    ),
    (
        # Previously broken patch format
        "ONNX_UNUSED( " + bs + nl + "      OpSchema(#name, __FILE__, __LINE__))",
        "[[maybe_unused]] static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(op_schema_register_once##name##Counter)(" + nl +
        "      ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(OpSchema(#name, __FILE__, __LINE__)))",
    ),
]

count = 0
for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        count += 1
        print(f"Applied replacement pattern {count}")

if count == 0:
    print("WARNING: No matching patterns found. Checking file content...")
    idx = text.find("ONNX_OPERATOR_SCHEMA_UNIQ")
    if idx >= 0:
        print(f"Found ONNX_OPERATOR_SCHEMA_UNIQ at {idx}")
        print(repr(text[idx:idx+200]))
else:
    schema_path.write_text(text)
    verify = schema_path.read_text()
    if "ONNX_UNUSED = \\" in verify or "ONNX_UNUSED( \\" in verify:
        print("ERROR: Patch verification failed!")
        exit(1)
    print(f"Patch verified successfully ({count} pattern(s) fixed).")
