# Trigpoint PostgreSQL extension (spike). PGXS builds trigpoint.so from the C module (trigpoint.c)
# plus libtrigpoint_core.a: the C++17 core (../csrc/cascade.cpp, hnswlib header-only) and the C shim
# (trigpoint_core.cpp), compiled -fPIC with the architecture flags csrc/build.sh uses
# (csrc/archflags.sh: -march=native on x86, NEON+dotprod on arm64). Built in docker/Dockerfile.pg's
# first stage; the compiler never runs on the host:
#     make && make install DESTDIR=/out
EXTENSION   = trigpoint
EXTVERSION  = 0.5
MODULE_big  = trigpoint
OBJS        = trigpoint.o tp_build.o tp_scan.o tp_shared.o tp_graph.o tp_pbuild.o tp_check.o
DATA        = trigpoint--0.2.sql trigpoint--0.2--0.3.sql trigpoint--0.3.sql trigpoint--0.3--0.4.sql trigpoint--0.4.sql \
              trigpoint--0.4--0.5.sql trigpoint--$(EXTVERSION).sql

CORE_DIR    = ../csrc
CORE_LIB    = libtrigpoint_core.a
CORE_OBJS   = core_cascade.o core_shim.o
CORE_REV   := $(shell sha256sum $(CORE_DIR)/cascade.cpp | cut -c1-12)
ARCHFLAGS  := $(shell sh $(CORE_DIR)/archflags.sh)
CXX        ?= g++
# SANITIZE=1: -O1 -g with AddressSanitizer + UBSan on the core and the C module, linked into
# trigpoint.so, built against the sanitized server of docker/Dockerfile.pg (PG_CONFIG points at it). Use gcc/g++.
SANITIZE   ?= 0
ifeq ($(SANITIZE),1)
SAN_FLAGS   = -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined -g
CORE_OPT    = -O1 $(SAN_FLAGS)
PG_CFLAGS   = $(SAN_FLAGS) -O1
else
SAN_FLAGS   =
CORE_OPT    = -O3
endif
CORE_CXXFLAGS = $(CORE_OPT) -std=c++17 $(ARCHFLAGS) -fPIC -fvisibility=hidden \
                -I$(CORE_DIR) -I$(CORE_DIR)/hnswlib \
                -DTRIGPOINT_CORE_REV='"$(CORE_REV)"' -DTRIGPOINT_ARCHFLAGS='"$(ARCHFLAGS)"'

PG_CPPFLAGS = -DTRIGPOINT_VERSION='"$(EXTVERSION)"' -I$(CORE_DIR)
# TEST_HOOKS=1: compile the page hooks trigpoint_debug_peek/poke and the result-changing
# diagnostics (debug_skip_heap, debug_no_delta_graph, allow_build_threads) in; never in a release build.
TEST_HOOKS ?= 0
ifeq ($(TEST_HOOKS),1)
PG_CPPFLAGS += -DTRIGPOINT_TEST_HOOKS
endif
SHLIB_LINK  = $(CORE_LIB) -lstdc++ -lm $(SAN_FLAGS)
EXTRA_CLEAN = $(CORE_LIB) $(CORE_OBJS)

PG_CONFIG  ?= pg_config
PGXS       := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

core_cascade.o: $(CORE_DIR)/cascade.cpp
	$(CXX) $(CORE_CXXFLAGS) -c -o $@ $<
core_shim.o: trigpoint_core.cpp trigpoint_core.h
	$(CXX) $(CORE_CXXFLAGS) -c -o $@ $<
$(CORE_LIB): $(CORE_OBJS)
	$(AR) rcs $@ $^
$(shlib): $(CORE_LIB)
trigpoint.o tp_build.o tp_scan.o tp_shared.o tp_graph.o tp_pbuild.o tp_check.o: trigpoint.h trigpoint_core.h tp_graph.h $(CORE_DIR)/flat_graph.h $(CORE_DIR)/flat_image.h $(CORE_DIR)/flat_check.h
