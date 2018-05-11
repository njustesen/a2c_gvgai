if which node > /dev/null 
then
    echo "Node is installed"
else
    if which apt-get > /dev/null 
    then
        curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
        sudo apt-get install -y nodejs
	sudo apt-get install npm
    else
        brew install node
    fi 
fi

npm install hilbert
npm install fs
