# To test flux, usually we build flux in its own build container and copy
# the whole repo to /opt/tiger. We may also install pre-built flux in /opt/tiger
export PYTHONPATH=/opt/tiger/flux/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/tiger/flux/lib:/opt/tiger/flux/build/lib:$LD_LIBRARY_PATH