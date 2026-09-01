# Third-party licenses

All vendored third-party assets in this repository are BSD-3-Clause licensed.
No GPL assets. Provenance (source repo, commit hash, file list) is recorded
in an `UPSTREAM` file next to each vendored directory:

| Directory | Upstream | License |
|---|---|---|
| `src/apollo_xarm7_sim/assets/ufactory_xarm7/` | google-deepmind/mujoco_menagerie @ `da76818e269b82289eba39808e2fb91d679d6994` | BSD-3-Clause (UFACTORY Inc.) |
| `src/apollo_xarm7_sim/assets/rail/` | M4D-SC1ENTIST/mavis_mujoco @ `ea233197700047a2f40af3e4817083504b153199` | BSD-3-Clause (inherited from menagerie) |
| `src/apollo_xarm7_sim/assets/cameras/` | M4D-SC1ENTIST/mavis_mujoco @ `ea233197700047a2f40af3e4817083504b153199` | BSD-3-Clause (inherited from menagerie) |

`ASSET_MANIFEST.json` (in `src/apollo_xarm7_sim/assets/`) records the sha256
of every vendored asset so recorded episodes can be replayed against a
verified asset set. Regenerate with
`uv run python -m apollo_xarm7_sim.tools.gen_asset_manifest`.

## BSD-3-Clause text (UFACTORY Inc.)

The identical license text ships as `LICENSE` inside each vendored
directory; reproduced here:

```
Copyright (c) 2018, UFACTORY Inc.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
