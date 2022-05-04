include ../lib/zfp-develop/Config

BINDIR = ../apps
TARGETS = $(BINDIR)/test
INCS = -I../lib/zfp-develop/include
LIBS = -L../lib/zfp-develop/lib -lzfp
CLIBS = $(LIBS) -lm
CXXLIBS = $(LIBS)

all: $(TARGETS)

$(BINDIR)/test: qsim_base.cc ../lib/zfp-develop/lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) -I../lib/zfp-develop/array qsim_base.cc $(CXXLIBS) -o $@
clean:
	-rm -f test