# Standalone build for custom BBV plugin with .disas output + exit flush.
# Uses qemu-plugin.h from the QEMU submodule (read-only reference).
#
# Usage:
#   make -C tools/bbv/          # build libbbv.so
#   make -C tools/bbv/ clean    # remove built artifacts

CC      ?= gcc
CFLAGS   := -shared -fPIC -O2 -Wall -Werror
CFLAGS  += $(shell pkg-config --cflags glib-2.0)
LDFLAGS  := $(shell pkg-config --libs glib-2.0)

QEMU_HDR = ../../third_party/qemu/include/qemu

libbbv.so: bbv.c $(QEMU_HDR)/qemu-plugin.h
	$(CC) $(CFLAGS) -I$(QEMU_HDR) -o $@ bbv.c $(LDFLAGS)

.PHONY: clean
clean:
	rm -f libbbv.so
