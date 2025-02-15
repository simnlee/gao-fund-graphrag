### Neo4j Database
There is a Neo4j AuraDB account already associated with associates@gao-cap.com. Just go to [the website](https://neo4j.com/cloud/platform/aura-graph-database/) and login with Google. The instance credentials are already in the .env file in the backend/ folder. If you have to update the instance details, just update the .env file. 

### Deployment
Alternatively, you can run the backend and frontend separately:

- For the frontend:
1. Create the frontend/.env file by copy/pasting the frontend/example.env.
2. Change values as needed
3.
    ```bash
    cd frontend
    yarn
    yarn run dev
    ```

- For the backend:
1. Create the backend/.env file by copy/pasting the backend/example.env.
2. Change values as needed
3.
    ```bash
    cd backend
    python -m venv envName
    source envName/bin/activate 
    pip install -r requirements.txt
    uvicorn score:app --reload
    ```
