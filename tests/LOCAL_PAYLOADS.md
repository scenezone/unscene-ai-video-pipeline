# Run the local episode data through the main API

Use these two independent files as the JSON body of `POST /v1/analyze/jobs`:

| Payload | Video | Clothing folder | Object folder | Products / images |
| --- | --- | --- | --- | --- |
| [payload_episode_1.json](payload_episode_1.json) | data/testdata/Episode 1.mp4 | data/productimages/clothing1 | data/productimages/objects1/by_product | 7 clothing references + 6 objects; 27 images |
| [payload_episode_2.json](payload_episode_2.json) | data/testdata/Episode 2.mp4 | data/productimages/clothing | data/productimages/objects/by_product | 36 clothing references + 7 objects; 60 images |

The actual directories are named **objects1** and **objects** (plural). Each object subdirectory is one product; its remaining images are `additionalImageUrls`. Clothing files are flat and have no verified product grouping, so each is a separate clothing reference. The main API detects clothing categories rather than matching their product IDs.

IDs use an episode prefix because OBJ002 is a handbag in Episode 1 and glasses in Episode 2. Every `imageKey` is the exact primary file path relative to `data/`. Spaces and parentheses in HTTP URLs are percent-encoded. [local_payload_mapping.csv](local_payload_mapping.csv) lists every image, product ID, primary/additional role and URL. These are local test IDs, not IDs imported from a live backend catalogue.

Object groups:

| Episode 1 ID | Directory | Views |
| --- | --- | --- |
| EP1_OBJ002 | OBJ002_handbag | 5 |
| EP1_OBJ003 | OBJ003_earrings | 1 |
| EP1_OBJ004 | OBJ004_eyeglasses | 4 |
| EP1_OBJ005 | OBJ005_phone | 4 |
| EP1_OBJ006 | OBJ006_keychain | 4 |
| EP1_OBJ007 | OBJ007_smartwatch | 2 |

| Episode 2 ID | Directory | Views |
| --- | --- | --- |
| EP2_OBJ002 | OBJ002_Glasses | 6 |
| EP2_OBJ003 | OBJ003_Phone | 3 |
| EP2_OBJ004 | OBJ004_Glass | 3 |
| EP2_OBJ005 | OBJ005_Bottle | 4 |
| EP2_OBJ006 | OBJ006_MetalVase | 3 |
| EP2_OBJ007 | OBJ007_WoodenTable | 3 |
| EP2_OBJ008 | OBJ008_smartwatch | 2 |

## 1. Serve the local files and receive callbacks

The API downloader requires HTTP(S) URLs, not Windows paths or file:// URLs. In a separate terminal, from the repository root:

```powershell
python tests/serve_local_assets.py
```

Leave this terminal running until both jobs finish. The helper serves only the two videos and 87 selected images on `http://127.0.0.1:8765`. It does not expose the authentication database, job database, data.zip, directory listings or arbitrary project files.

The helper also accepts POST `/callback`, saves the received JSON to `outputs/local_callbacks/<jobId>.json`, and returns HTTP 204. Default result paths:

```text
outputs/local_callbacks/episode-1-local-v1.json
outputs/local_callbacks/episode-2-local-v1.json
```

No external webhook is needed. No real purchase URLs were supplied: `purchaseLink` intentionally points to the local primary-image preview for these tests. Replace it with a real shop URL for backend use. The helper only serves test media and receives test callbacks; it does not start the AI API or run inference.

## 2. Start and authorize the main API

In your existing configured AI environment, start the main API and its model dependencies as usual, for example:

```powershell
python main.py
```

Keep Ollama running with `qwen2.5:3b-instruct` available for clothing prompt cleanup. The existing GroundingDINO/SAM weights and CLIP/Florence dependencies must be available. The payloads do not replace model/environment setup.

Open `http://127.0.0.1:8000/docs`, select **Authorize**, and enter your API client ID and secret as the Basic username/password. Open `POST /v1/analyze/jobs`, click **Try it out**, and paste one payload file's contents. Submit each file as a separate request. The API client needs `jobs:write` to submit and `jobs:read` to poll. No credentials are stored in these payload files.

Alternatively, PowerShell can prompt for credentials without writing them into the JSON:

```powershell
$apiCredential = Get-Credential -Message "Enter the main API client ID and secret"
$basicValue = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes(
        $apiCredential.UserName + ":" + $apiCredential.GetNetworkCredential().Password
    )
)
$apiHeaders = @{ Authorization = "Basic " + $basicValue }

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/analyze/jobs" `
    -Headers $apiHeaders -ContentType "application/json" `
    -InFile "tests/payload_episode_1.json"

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/analyze/jobs" `
    -Headers $apiHeaders -ContentType "application/json" `
    -InFile "tests/payload_episode_2.json"
```

## 3. Poll the saved results

Using the same PowerShell session and headers:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/analyze/jobs/episode-1-local-v1" -Headers $apiHeaders
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/analyze/jobs/episode-2-local-v1" -Headers $apiHeaders
```

Accepted means queued, not completed. The single local inference worker processes the two jobs sequentially. The callback files appear when results are delivered, including a failure result if processing fails. If a callback fails because the helper was stopped, restart it and use the authenticated callback-retry endpoint; saved inference results do not need to be recomputed.

## 4. If the API runs in Docker or on another machine

`127.0.0.1` inside the AI container/remote host points to that environment, not the Windows machine holding these files. The two checked-in payloads default to a same-machine API. For Docker Desktop, run the helper listening on a reachable interface:

```powershell
python tests/serve_local_assets.py --bind 0.0.0.0
```

Then regenerate the URLs for the AI's network view:

```powershell
python tests/prepare_local_payloads.py --media-base-url "http://host.docker.internal:8765" --run-tag docker-v1
```

For a remote AI deployment, use a reachable hostname/IP for the machine serving the media, or upload these exact files to accessible storage and use that storage's base URL. A remote host cannot download your local files from localhost. A cloud server generally cannot reach a private laptop LAN address without a network route. Set `--callback-url` separately when results should go to your existing backend receiver.

```powershell
python tests/prepare_local_payloads.py --media-base-url "http://YOUR-REACHABLE-HOST:8765" `
    --callback-url "https://YOUR-BACKEND/ai/results" --run-tag remote-v1
```

Replace example hostnames before use. These commands overwrite only the generated payloads and mapping CSV, preserving input files and existing hand-written payload examples. Prefer the loopback binding for same-machine tests; the helper has no authentication.

## 5. Regenerate or run fresh inference

```powershell
python tests/prepare_local_payloads.py
python tests/prepare_local_payloads.py --run-tag local-v2
```

The generator scans the current episode folders and asserts that every selected image is assigned exactly once. Grouped object IDs remain stable. Clothing IDs use filename plus a stable path hash. Reusing the same job ID and body reuses the saved job; changing a body under an existing ID returns 409. Use a fresh run tag for a new inference run or changed host/asset configuration.

The original `modified_payload.json` and `payload_with_imagekey.json` have not been changed. No job is automatically submitted, no credentials are rotated, and the existing main pipeline code is unchanged.
