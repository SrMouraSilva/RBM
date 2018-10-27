#!/bin/bash
#if [ -d "venv" ]; then
#    source venv/bin/activate
#fi


if [ "$1" = 'init' ]; then
    echo ">> Initializing environment"

    if [ ! -d "venv" ]; then
        echo ">> Creating venv"
        python -m venv venv
    fi
    #echo "source $(pwd)/venv/bin/activate"

    echo ">> Installing dependencies"
    
    source venv/bin/activate
    python setup.py develop
    
    echo ">> Finished!"

elif [ "$1" = 'python' ]; then
    source venv/bin/activate
    python ${@:2}

elif [ "$1" = 'bash' ]; then
    source venv/bin/activate
    execute python
    echo "$@"
    bash
    #bash "$@"
fi
