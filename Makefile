CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -pthread
# Check if system has native jpeglib.h installed (prefer system headers to match installed libjpeg)
ifeq ($(wildcard /usr/include/jpeglib.h),)
    INCLUDES = -Iinclude
    LIBJPEG_VER62 := $(shell ldconfig -p 2>/dev/null | grep -E "libjpeg\.so\.62" | head -n 1)
    ifneq ($(LIBJPEG_VER62),)
        CFLAGS += -DJPEG_LIB_VERSION=62
    endif
else
    INCLUDES =
endif

# Detect architecture for Raspberry Pi Zero 2 tuning
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M), aarch64)
    # Raspberry Pi Zero 2 running 64-bit OS (ARM Cortex-A53)
    ARCH_FLAGS = -march=armv8-a+crc -mtune=cortex-a53
else ifneq ($(filter armv7% armhf, $(UNAME_M)),)
    # Raspberry Pi Zero 2 running 32-bit OS
    ARCH_FLAGS = -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
else
    ARCH_FLAGS =
endif

CFLAGS += $(ARCH_FLAGS)

# Find libjpeg library location
LIBJPEG_SO := $(shell ldconfig -p 2>/dev/null | grep -E "libjpeg\.so(\.[0-9]+)?" | head -n 1 | awk '{print $$NF}')
ifeq ($(LIBJPEG_SO),)
    LIBS = -ljpeg -lpthread -lm
else
    LIBS = $(LIBJPEG_SO) -lpthread -lm
endif

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin

TARGET = fastcrop
SRCS = src/fastcrop.c

.PHONY: all clean install uninstall test

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) $(LIBS) -o $(TARGET)
	@echo "Build successful: $(TARGET)"

clean:
	rm -f $(TARGET)
	rm -rf test_out /tmp/test_real /tmp/test_crops_*

install: $(TARGET)
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 $(TARGET) $(DESTDIR)$(BINDIR)/$(TARGET)
	install -m 755 fastcrop.py $(DESTDIR)$(BINDIR)/fastcrop-py
	install -m 755 fastdetect.py $(DESTDIR)$(BINDIR)/fastdetect

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/$(TARGET)
	rm -f $(DESTDIR)$(BINDIR)/fastcrop-py
	rm -f $(DESTDIR)$(BINDIR)/fastdetect

test: $(TARGET)
	@python3 benchmark.py --quick
