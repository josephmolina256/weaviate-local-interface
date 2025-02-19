# weaviate-local-interface
Toolbox of APIs for working with local Weaviate docker instance

Prerequisites:
Python
Docker

To set up your weaviate container, we're going to be using this basic pre-set docker-compose.yml file that is for a local weaviate instance where we will be providing our own embeddings/vectors. This means NO models inside! Feel free to edit the file as desired. Here is some documentation that might help with that:
https://weaviate.io/developers/weaviate/installation/docker-compose

Our default config is set to save data to a ~/Documents/weaviate_data folder on your local machine. You may need to create that folder yourself and you can always move it by changing the docker-compose.yml.

Once you've settled on a docker compose config, run the following:
```
docker compose up -d
```
Remove the -d flag if you want to have the container open on a terminal to see inside.

If you want to stop the container, please refrain from just ctrl+c or closing the terminal and instead run docker compose down to ensure safe data storage.
```
docker compose down
```

Next, onto running our FastAPI interface.

Create and activate virtual environment.
```
python -m venv .venv
source .venv/Scripts/activate 
#.venv/Scripts/activate for powershell or command prompt
#.venv/Scripts/activate.bat otherwise
```
You should see a little (.venv) tag in your terminal now.

Run the following for installation requirements
```
pip install -r requirements.txt
```

From there, to run the endpoint, run this command:
```
python -m app.inference
```