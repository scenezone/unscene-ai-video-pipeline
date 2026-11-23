clientId = "cli_A3griGT5lWZaopl8"
clientSecret = "x38nY8wfGzJIx32ZHVKV_y2bbXzeC3mzEhd9GKqr3Zk"


python -m services.auth_admin create --name "Partner A" --scopes "jobs:write jobs:read"
python -m services.auth_admin create --name "Partner B" --scopes "jobs:write jobs:read"

python -m services.auth_admin revoke <that-users-client-id>