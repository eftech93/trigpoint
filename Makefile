# Top-level Makefile of the release tree, for PGXN (`pgxn install trigpoint`) and plain source installs:
#     make && sudo make install          # PG_CONFIG=/path/to/pg_config to pick a server
# The extension's own Makefile is pgext/Makefile; the C++ core is in csrc/ (hnswlib is bundled in the PGXN zip;
# from a git checkout, clone it first: git clone https://github.com/nmslib/hnswlib csrc/hnswlib).
# The build tunes for this machine (-march=native); pass ARCHFLAGS=-march=x86-64-v3 for a portable x86-64 build.
.PHONY: all install installcheck uninstall clean
all install installcheck uninstall clean:
	$(MAKE) -C pgext $@
