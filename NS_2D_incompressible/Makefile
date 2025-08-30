# Makefile à la racine du projet

.PHONY: configure build clean rebuild help

# 1) configure crée (ou met à jour) build/ sans tout supprimer
configure:
	mkdir -p build
	cd build && cmake ..
	cd build && make

# 2) build appelle make dans build/ — Make y recompile seulement ce qui a changé
build: configure
	$(MAKE) -C build

# 3) clean ne détruit pas tout, il fait juste clean dans build/
clean:
	$(MAKE) -C build clean

# 4) rebuild = clean + build
rebuild: clean build

# 5) help pour lister les cibles
help:
	@echo "Usage:"
	@echo "  make build     # compile (incrémental)"
	@echo "  make clean     # supprime les .o, mais garde build/ cmake intact"
	@echo "  make rebuild   # clean + build"
	@echo "  make help      # ce message"