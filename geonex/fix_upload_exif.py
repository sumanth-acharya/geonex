"""
Script to update upload.html:
1. Replace location input section with EXIF-GPS-aware version
2. Insert exifr.js script + file-change handler before </body>
"""

with open('templates/upload.html', 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File length: {len(content)} chars")

# -----------------------------------------------
# 1. Replace the location section
# -----------------------------------------------

# Try to find the old section (could be either version)
# First check what's currently there
if 'placeholder="Enter location (e.g. Hyderabad) or auto-detect' in content:
    # Has the auto-detect button version, replace it
    import re
    # Find the location div block
    pattern = r'<!-- Location Input with Auto-Detect -->(.*?)<!-- Image Upload -->'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        old_block = match.group(0)
        new_block = '''<!-- Location Input (EXIF GPS auto-fill) -->
                  <div class="col-lg-12">
                    <div class="input-group">
                      <label>Location Name <span id="geoTag" style="font-size:11px; color:#00c3ff; font-weight:normal; display:none;">&#128205; Auto-detected from image GPS</span></label>
                      <input
                        type="text"
                        name="location"
                        id="locationInput"
                        placeholder="Location auto-detects from image GPS — or enter manually"
                        required
                        style="width:100%;"
                      />
                      <small id="geoStatus" style="display:none; margin-top:6px; font-size:12px;"></small>
                    </div>
                  </div>
                  <!-- Image Upload -->'''
        content = content.replace(old_block, new_block)
        print("Location section: REPLACED (auto-detect button version)")
    else:
        print("WARNING: Could not find location section with regex")

elif 'placeholder="Enter location name (e.g. Hyderabad)' in content:
    # Has the original plain version
    old_loc = '''                  <!-- Location Input -->
                  <div class="col-lg-12">
                    <div class="input-group">
                      <label>Location Name</label>
                      <input type="text" name="location" placeholder="Enter location name (e.g. Hyderabad)" required />
                    </div>
                  </div>'''
    new_loc = '''                  <!-- Location Input (EXIF GPS auto-fill) -->
                  <div class="col-lg-12">
                    <div class="input-group">
                      <label>Location Name <span id="geoTag" style="font-size:11px; color:#00c3ff; font-weight:normal; display:none;">&#128205; Auto-detected from image GPS</span></label>
                      <input
                        type="text"
                        name="location"
                        id="locationInput"
                        placeholder="Location auto-detects from image GPS — or enter manually"
                        required
                        style="width:100%;"
                      />
                      <small id="geoStatus" style="display:none; margin-top:6px; font-size:12px;"></small>
                    </div>
                  </div>'''
    if old_loc in content:
        content = content.replace(old_loc, new_loc)
        print("Location section: REPLACED (original version)")
    else:
        print("WARNING: Original location section not found verbatim")
else:
    print("WARNING: Neither location version found — check file manually")

# -----------------------------------------------
# 2. Remove any old autoDetectLocation script block
# -----------------------------------------------
import re
old_script_pattern = r'<!-- ===== Auto-Detect Location =====.*?</script>'
content, n = re.subn(old_script_pattern, '', content, flags=re.DOTALL)
if n:
    print(f"Old auto-detect script: REMOVED ({n} occurrence(s))")

# -----------------------------------------------
# 3. Insert new exifr.js + EXIF GPS script before </body>
# -----------------------------------------------
exif_script = '''
  <!-- exifr: reads EXIF GPS coordinates embedded in uploaded image files -->
  <script src="https://cdn.jsdelivr.net/npm/exifr@7.1.3/dist/lite.umd.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function () {
      var fileInput = document.querySelector('input[type="file"][name="image"]');
      var locInput  = document.getElementById('locationInput');
      var geoStatus = document.getElementById('geoStatus');
      var geoTag    = document.getElementById('geoTag');

      if (!fileInput || !locInput) return;

      fileInput.addEventListener('change', function () {
        var file = fileInput.files[0];
        if (!file) return;

        // Reset status
        if (geoTag) geoTag.style.display = 'none';
        geoStatus.style.display = 'block';
        geoStatus.style.color = '#00c3ff';
        geoStatus.textContent = 'Reading GPS data from image file...';

        // exifr.gps() reads the GPS EXIF tags from the selected file
        exifr.gps(file).then(function (gps) {
          if (!gps || gps.latitude === undefined || gps.latitude === null) {
            geoStatus.style.color = '#94a3b8';
            geoStatus.textContent = 'No GPS data found in this image. Please enter the location manually.';
            return;
          }

          var lat = gps.latitude;
          var lon = gps.longitude;
          geoStatus.textContent = 'GPS found (' + lat.toFixed(5) + ', ' + lon.toFixed(5) + ') — fetching location name...';

          // Free reverse-geocoding via OpenStreetMap Nominatim (no API key needed)
          fetch(
            'https://nominatim.openstreetmap.org/reverse?format=json&lat=' + lat + '&lon=' + lon + '&zoom=10&accept-language=en',
            { headers: { 'User-Agent': 'GEONEX/1.0' } }
          )
          .then(function (res) { return res.json(); })
          .then(function (data) {
            var a     = data.address || {};
            var city  = a.city || a.town || a.village || a.county || '';
            var state = a.state || '';
            var ctry  = a.country || '';
            var parts = [];
            if (city) parts.push(city);
            if (state && state !== city) parts.push(state);
            if (ctry)  parts.push(ctry);
            var name = parts.join(', ') || data.display_name || (lat.toFixed(4) + ', ' + lon.toFixed(4));

            locInput.value = name;
            if (geoTag) geoTag.style.display = 'inline';
            geoStatus.style.color = '#00ff9c';
            geoStatus.textContent = 'Location auto-detected from image: ' + name;
          })
          .catch(function () {
            // GPS found but geocoding failed — use raw coordinates
            locInput.value = lat.toFixed(5) + ', ' + lon.toFixed(5);
            geoStatus.style.color = '#fbbf24';
            geoStatus.textContent = 'GPS found but name lookup failed. Using coordinates: ' + locInput.value;
          });

        }).catch(function () {
          geoStatus.style.color = '#f87171';
          geoStatus.textContent = 'Could not read image GPS data. Please enter location manually.';
        });
      });
    });
  </script>
'''

if '</body>' in content:
    content = content.replace('</body>', exif_script + '\n</body>', 1)
    print("EXIF GPS script: INSERTED before </body>")
else:
    print("ERROR: </body> not found in file!")

with open('templates/upload.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("File saved successfully.")
print(f"New file length: {len(content)} chars")
