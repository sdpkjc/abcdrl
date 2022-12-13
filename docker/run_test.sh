#!/bin/bash

if [[ "$SOURCE_BRANCH" =~ "main" ]] ; then
    pytest
else
    echo "$SOURCE_BRANCH branch skip test."
fi
