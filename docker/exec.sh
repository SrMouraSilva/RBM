#!/bin/bash
BOLD='\033[1m'
NORMAL='\033[0m'

execute() {
    sudo docker run -it --rm \
      -v "$PWD":/rbm \
      -v "$PWD/docker/bash.bashrc":/etc/bash.bashrc \
      -w /rbm \
      -u $(id -u):$(id -g) \
      python:3-slim-stretch "$@"
      
}

if [ "$1" = 'tf' ]; then
    sudo docker run -it --rm \
      -v "$PWD":/rbm \
      -p 8888:8888 \
      -u $(id -u):$(id -g) \
      -w /rbm \
      tensorflow/tensorflow:latest-py3 \
      ${@:2}

elif [ "$1" = 'python' ]; then
    #execute python
    execute bash docker/entrypoint.sh python
    #execute /bin/bash -c "python"
    #$@

elif [ "$1" = 'bash' ]; then
    execute bash
    #execute bash -c "${@:2}"

elif [ "$1" = 'init' ]; then
    execute bash docker/entrypoint.sh init
    
elif [ "$1" = 'help' ]; then
    printf ""
	printf "Commands\n"
	printf "    ${BOLD}init${NORMAL}\n"
	printf "          Initialize the environment workspace\n"
	printf "    ${BOLD}bash${NORMAL}\n"
	printf "          bash console (/bin/bash)\n"
	printf "    ${BOLD}python${NORMAL}\n"
	printf "          python console\n"
	#printf "    ${BOLD}clean${NORMAL}\n"
	#printf "          Clean files\n"
	#printf "    ${BOLD}docs${NORMAL}\n"
	#printf "          Make the docs\n"
	#printf "    ${BOLD}docs-see${NORMAL}\n"
	#printf "          Make the docs and open it in BROWSER\n"
	#printf "    ${BOLD}install-docs-requirements${NORMAL}\n"
	#printf "          Install the docs requirements\n"
	#printf "    ${BOLD}install-tests-requirements${NORMAL}\n"
	#printf "          Install the tests requirements\n"
	#printf "    ${BOLD}test${NORMAL}\n"
	#printf "          Execute the tests\n"
	#printf "    ${BOLD}test-details${NORMAL}\n"
	#printf "          Execute the tests and shows the result in BROWSER\n"
	#printf "           - BROWSER=firefox\n"
	#printf "    ${BOLD}help${NORMAL}\n"
	#printf "          Show the valid commands\n"

else
    execute "$@"
fi
