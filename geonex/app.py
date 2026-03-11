from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import re
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
from datetime import date
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

IMG_SIZE = (224, 224)
MODEL_PATH = "Models/mobilenetv2.h5"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# Replace with YOUR actual class list from training output
CLASS_NAMES = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

# Dictionary defining detailed technical analysis for land cover classes
DETAILED_REPORTS = {
    'airplane': '[SPECTRAL SIGNATURE]\n- VIS reflectance: High on metallic fuselage surfaces\n- NIR reflectance: High (aluminium/composite specular)\n- Thermal: Low emissivity (metallic surface)\n- Shadow: Elongated bilateral -- diagnostic of fixed-wing geometry\n\n[SPATIAL METRICS]\n- Object length: 20-80m | Wingspan: 15-65m (GSD-normalised)\n- Aspect ratio: ~1.3:1 (fuselage:wingspan)\n- Texture: Smooth, specular reflection on tarmac\n- Shape: Bilateral symmetry -- strong discriminator vs ground vehicles\n\n[CLASSIFICATION FINDINGS]\n- Class: Airplane (object-level detection, airport/airfield context)\n- Model: MobileNetV2 softmax probability output\n- Training similarity: NWPU-RESISC45 airplane tiles (GSD 0.2-30m)\n- Contextual cues: Tarmac, taxiway markings, apron proximity\n\n[ENVIRONMENTAL INDICATORS]\n- LST: Elevated (reflective tarmac + engine heat)\n- ISC: >95% in surrounding apron zone\n- NDVI: ~0.0 (no vegetation)\n\n[APPLICATIONS]\nSAR/optical fusion for aircraft detection | Air-traffic density monitoring | Airport capacity planning | Defense intelligence | Emergency airlift coordination',
    'airport': '[SPECTRAL SIGNATURE]\n- Runway surface: Grey asphalt, moderate VIS reflectance (R~0.15, G~0.14, B~0.13)\n- Terminal rooftops: High NIR reflectance (metal/concrete)\n- NDVI: ~0.0-0.05 (dominantly impervious surface)\n- Thermal: Elevated LST across runway and apron surfaces\n\n[SPATIAL METRICS]\n- Runway: 1,800-4,500m length x 30-60m width\n- Terminal footprint: 10,000-200,000 m2\n- Runway orientation: Aligned with prevailing wind (ILS/VOR indicator)\n- Texture: Fine on runways (painted markings), coarse on surrounding zones\n\n[CLASSIFICATION FINDINGS]\n- Class: Airport (facility-level land-use classification)\n- Key discriminators: Parallel runway geometry, apron-terminal adjacency\n- Training match: High spatial frequency linear features + low NDVI\n- False positive risk: Industrial zones (similar ISC) -- differentiated by runway geometry\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: >80% (major impervious zone)\n- Emissions: NOx/VOC point sources (combustion)\n- Stormwater risk: High (glycol de-icing, jet fuel runoff)\n- UHI contribution: Significant\n\n[APPLICATIONS]\nAirport expansion monitoring | Runway capacity analysis | Noise contour mapping | Environmental compliance (ICAO Annex 16) | Emergency runway assessment',
    'baseball_diamond': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Baseball Diamond\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Baseball Diamond (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Baseball Diamond training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'basketball_court': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Basketball Court\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Basketball Court (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Basketball Court training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'beach': '[SPECTRAL SIGNATURE]\n- Sandy substrate: Very high VIS reflectance (R~0.55, G~0.50, B~0.45)\n- Shoreline boundary: Sharp spectral contrast (sand ~0.5 vs water ~0.02-0.05)\n- NDWI (McFeeters): Positive in intertidal zone; negative on dry sand\n- NIR: Low (non-vegetated) -- distinguishes from dune grasslands\n\n[SPATIAL METRICS]\n- Shoreline sinuosity index: 1.0-1.3 (linear to mildly curved)\n- Beach width (GSD-relative): 5-300m\n- Texture: Low spatial frequency (smooth sand surface)\n- Wet/dry sand boundary: Detectable at <5m GSD\n\n[CLASSIFICATION FINDINGS]\n- Class: Beach (coastal sedimentary depositional environment)\n- Key discriminators: High-albedo strip adjacent to dark water body\n- Confusion risk: Dry riverbeds (similar spectral) -- differentiated by water adjacency\n- Training match: Strong NWPU-RESISC45 beach class alignment\n\n[ENVIRONMENTAL INDICATORS]\n- Shoreline Change Rate: Detectable via multi-temporal NDWI/DSAS\n- Coastal Vulnerability Index (CVI): Moderate-High\n- Sea Level Rise sensitivity: High (low-elevation depositional zone)\n- Nesting habitat: Cheloniidae, Charadriiformes (protected)\n\n[APPLICATIONS]\nShoreline change detection (DSAS) | Coastal erosion risk modelling | MPA delineation | Storm surge impact | Tourism carrying capacity',
    'bridge': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Bridge\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Bridge (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Bridge training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'chaparral': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Chaparral\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Chaparral (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Chaparral training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'church': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Church\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Church (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Church training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'circular_farmland': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Circular Farmland\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Circular Farmland (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Circular Farmland training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'cloud': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Cloud\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Cloud (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Cloud training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'commercial_area': '[SPECTRAL SIGNATURE]\n- Flat rooftops: Bright NIR (metal/EPDM) or moderate VIS (concrete/gravel)\n- NDVI: 0.0-0.1 (near-zero vegetation)\n- ISC: >85% (near-complete imperviousness)\n- LST: Very high daytime (reflective/absorptive flat roofs + HVAC waste heat)\n- Parking surfaces: Dark asphalt (low VIS reflectance 0.05-0.10)\n\n[SPATIAL METRICS]\n- Building footprint: Large (1,000-50,000 m2)\n- Roof-to-ground ratio: 0.5-0.8\n- Texture: Coarse within buildings, fine in parking zones\n- Road network: Wide arterials (8-30m) with loading bays\n\n[CLASSIFICATION FINDINGS]\n- Class: Commercial/Retail Zone (Urban Built-up B2 -- LCCS)\n- Key discriminators: Large homogeneous rooftops + expansive parking + road access\n- False positive: Industrial (similar ISC) -- differentiated by loading dock geometry\n\n[ENVIRONMENTAL INDICATORS]\n- LST: 5-10 deg C above vegetated land (peak afternoon)\n- Energy consumption: High per m2 (HVAC, lighting -- CO2 equivalent high)\n- Stormwater: Near-total impervious runoff (>0.85 runoff coefficient)\n\n[APPLICATIONS]\nRetail trade area mapping | UHI hotspot analysis | EV charging infrastructure siting | Commercial property assessment | Transport demand modelling',
    'dense_residential': '[SPECTRAL SIGNATURE]\n- Rooftop mix: Red tile (high R, low B) | Asphalt (low VIS) | Concrete (flat moderate VIS)\n- NDVI: 0.0-0.15 (minimal vegetation between structures)\n- ISC: >75% (high impervious surface cover)\n- LST: High UHI hotspot (4-8 deg C above suburban fringe)\n- Shadow fraction: High (multi-storey structures, narrow lanes)\n\n[SPATIAL METRICS]\n- Building density: >60 units/ha\n- Plot coverage ratio: 0.7-0.9\n- Road lane width: 3-8m (narrow lanes)\n- Texture: Fine-grain, high spatial frequency\n\n[CLASSIFICATION FINDINGS]\n- Class: Dense Residential (Urban Built-up B1.1 -- LCCS)\n- Key discriminators: High rooftop density + fine texture + low NDVI\n- Differentiated from commercial: Smaller footprint size, denser packing pattern\n- Morphological type: Organic grid (informal) vs. regular grid (planned)\n\n[ENVIRONMENTAL INDICATORS]\n- UHI Intensity: High (>4 deg C above rural reference)\n- Green Space Ratio: <10% (below WHO 9m2/person recommendation)\n- Population density proxy: High (>10,000 persons/km2)\n- Flood runoff coefficient: 0.7-0.9 (high imperviousness)\n\n[APPLICATIONS]\nPopulation density modelling | Slum/informal settlement mapping | UHI mitigation planning | Infrastructure demand analysis | Census augmentation',
    'desert': '[SPECTRAL SIGNATURE]\n- Bare soil: High VIS reflectance (R~0.35-0.60, spectrally flat to red-shifted)\n- NDVI: <0.1 (near-zero or negative -- bare mineral surface)\n- Albedo: 0.25-0.45 (sand) to 0.10-0.20 (dark reg/volcanic)\n- Mineralogy: Iron oxide -> red/orange hue | Quartz -> bright white\n- LST daytime: Very high (50-70 deg C surface temperature)\n\n[SPATIAL METRICS]\n- Surface roughness: Low (smooth ergs) to high (rocky reg/yardangs)\n- Dune wavelength: 50m-2km (detectable at moderate GSD)\n- Texture: Directional (wind-formed dune NE orientation dominant)\n- Sand encroachment front: Advancing bright spectral boundary\n\n[CLASSIFICATION FINDINGS]\n- Class: Desert/Hyper-arid Zone (Koppen BWh/BWk)\n- Key discriminators: High albedo + near-zero NDVI + no structural features\n- Sub-types: Erg (dunes) | Reg (gravel plain) | Hammada (rocky plateau)\n- Confusion risk: Dry riverbeds, bare agricultural soil -- context differentiates\n\n[ENVIRONMENTAL INDICATORS]\n- Rainfall: <250mm/yr (hyper-arid: <25mm/yr)\n- Desertification rate: Monitorable via NDVI trend and albedo change analysis\n- Dust emission potential: High (Aeolian PM10/PM2.5 source)\n- LST diurnal range: 30-50 deg C (low thermal inertia)\n\n[APPLICATIONS]\nDesertification monitoring (UNCCD) | CSP/PV site selection | Mineral exploration | Dune migration modelling | Dust storm source attribution',
    'forest': '[SPECTRAL SIGNATURE]\n- NDVI: 0.65-0.90 (dense, healthy broadleaf/mixed canopy)\n- NIR reflectance: Very high (0.45-0.55) -- cell structure scattering\n- Red band: Strong absorption (0.04-0.08) -- chlorophyll a/b\n- EVI (Enhanced Vegetation Index): 0.4-0.7\n- NDWI (Gao): 0.1-0.4 (canopy water content indicator)\n- SWIR: Moderate -- canopy moisture stress indicator\n\n[SPATIAL METRICS]\n- Canopy closure: >75%\n- Texture: High spatial variance (crown shadow heterogeneity)\n- Fractal dimension: 1.6-1.9 (complex edge geometry)\n- Crown diameter (GSD-dependent): 3-25m detectable\n\n[CLASSIFICATION FINDINGS]\n- Class: Forest -- Dense Canopy Land Cover (LCCS Class A1.1)\n- Training similarity: NWPU-RESISC45 forest tiles (strong spectral-spatial match)\n- Confusion risk: Chaparral (lower NDVI) | Meadow (no tree shadow texture)\n- CNN features extracted: Edge gradients, crown texture, shadow pattern\n\n[ENVIRONMENTAL INDICATORS]\n- Carbon Stock: 80-300 tC/ha (biome-dependent)\n- Above-Ground Biomass (AGB): Very High (SAR/LiDAR for precise estimate)\n- LST suppression: 3-8 deg C cooler than surrounding non-vegetated land\n- Evapotranspiration: High (latent heat flux dominant)\n- Fire Risk Index: Moderate (moisture-dependent)\n\n[APPLICATIONS]\nREDD+ carbon credit verification | AGB mapping | Deforestation alert (GLAD/PRODES) | Wildfire fuel load assessment | Biodiversity hotspot identification | Watershed zoning',
    'freeway': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Freeway\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Freeway (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Freeway training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'golf_course': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Golf Course\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Golf Course (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Golf Course training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'ground_track_field': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Ground Track Field\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Ground Track Field (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Ground Track Field training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'harbor': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Harbor\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Harbor (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Harbor training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'industrial_area': '[SPECTRAL SIGNATURE]\n- Metal roofs: Very high NIR reflectance (specular) + moderate VIS\n- Stockpile materials: Spectral mix of coal (black), limestone (white), ore (brown)\n- NDVI: ~0.0 (no vegetation in active industrial zone)\n- Thermal: Elevated point sources (kiln, furnace, cooling towers visible in TIR)\n- Smoke plumes: Detectable in TIR/SWIR bands\n\n[SPATIAL METRICS]\n- Building footprint: Very large (5,000-200,000 m2), low pitch roofs\n- ISC: >80%\n- Road access: Wide (HGV routing) with rail siding in heavy industry\n- Chimney shadows: Point source indicators (height from shadow length)\n\n[CLASSIFICATION FINDINGS]\n- Class: Industrial/Manufacturing Zone (B3 -- LCCS)\n- Key discriminators: Large-footprint buildings + material stockpiles + rail/road access\n- Pollution pressure: Chimney plume direction indicates prevailing wind vector\n\n[ENVIRONMENTAL INDICATORS]\n- Air pollution (PM2.5, SO2, NOx): detectable via Sentinel-5P TROPOMI\n- Groundwater contamination risk: High (heavy metals, solvents, leachate)\n- LST anomaly: Point source thermal signature visible in TIRS band\n\n[APPLICATIONS]\nIndustrial emission attribution | Environmental compliance monitoring | Supply chain infrastructure mapping | Land contamination risk screening',
    'intersection': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Intersection\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Intersection (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Intersection training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'island': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Island\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Island (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Island training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'lake': '[SPECTRAL SIGNATURE]\n- Open water reflectance: Very low NIR (<0.05) -- strong absorption\n- NDWI: >0.3 (clean freshwater); reduced in turbid/eutrophic conditions\n- Turbidity index: Elevated red/green ratio (TSS proxy)\n- Cyanobacterial bloom: NIR shoulder at 700nm (MCI index detectable)\n- Depth indicator: Blue/green ratio (clearer water = higher blue fraction)\n\n[SPATIAL METRICS]\n- Shape index: Circular (volcanic) to elongate (glacial/tectonic origin)\n- Shoreline sinuosity: 1.0-2.5\n- Surface area: Direct measurement via NDWI thresholding\n- Fetch length: Wind-wave energy estimation key variable\n\n[CLASSIFICATION FINDINGS]\n- Class: Inland Water Body -- Lake (LCCS W1.1)\n- Key discriminators: Enclosed dark water mass with defined land boundary\n- Sub-types: Freshwater | Saline (elevated blue/green) | Seasonal (NDWI time-series)\n- Confusion risk: Large ponds, reservoirs -- size threshold differentiation\n\n[ENVIRONMENTAL INDICATORS]\n- Trophic state: Oligotrophic to Eutrophic (NDVI/Chl-a remote sensing proxies)\n- Storage volume change: Surface area x altimetry-derived bathymetry\n- Drought sensitivity: High -- area loss detectable in multi-temporal analysis\n- Methane source: Anaerobic sediment degassing (eutrophic lakes)\n\n[APPLICATIONS]\nWater quality monitoring (WHO compliance) | Reservoir storage estimation | Algal bloom early warning | Drought assessment | Ramsar wetland status',
    'meadow': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Meadow\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Meadow (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Meadow training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'medium_residential': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Medium Residential\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Medium Residential (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Medium Residential training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'mobile_home_park': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Mobile Home Park\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Mobile Home Park (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Mobile Home Park training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'mountain': '[SPECTRAL SIGNATURE]\n- Rock surfaces: Variable reflectance (0.10-0.45) -- lithology dependent\n- Snow/ice cap: Very high albedo (0.60-0.90) -- dominates upper elevations\n- Vegetation belts: NDVI gradient 0.6-0.8 (montane) -> 0.2-0.4 (alpine) -> ~0 (barren)\n- Shadow fraction: High (steep illumination angle at high relief)\n- Thermal: Low LST at altitude (lapse rate ~6.5 deg C/1000m)\n\n[SPATIAL METRICS]\n- Relief amplitude: Hundreds to thousands of metres\n- Slope gradient: Inferred from shadow azimuth and DEM derivatives\n- Texture: Very high spatial variance (heterogeneous surface composition)\n- Ridge geometry: High fractal dimension (crenulated boundary pattern)\n\n[CLASSIFICATION FINDINGS]\n- Class: Mountain/Highland Terrain (Complex terrain -- LCCS)\n- Key discriminators: Shadow gradient + snow cap + vegetation zonation\n- Altitudinal zones: Montane / Subalpine / Alpine / Nival\n- Confusion risk: Chaparral at foothills -- differentiated by shadow intensity\n\n[ENVIRONMENTAL INDICATORS]\n- Snow Cover Area (SCA): Detectable via NDSI index\n- Glacier mass balance: Terminus retreat measurable via multi-temporal imagery\n- Landslide susceptibility: High on slopes >30 degrees\n- Permafrost: Detectable via thermal anomaly mapping\n\n[APPLICATIONS]\nGlacier mass balance (GRACE/ICESat) | Snowmelt runoff modelling | Landslide hazard mapping | Alpine biodiversity surveys | Avalanche risk zoning',
    'overpass': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Overpass\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Overpass (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Overpass training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'palace': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Palace\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Palace (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Palace training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'parking_lot': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Parking Lot\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Parking Lot (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Parking Lot training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'railway': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Railway\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Railway (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Railway training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'railway_station': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Railway Station\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Railway Station (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Railway Station training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'rectangular_farmland': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Rectangular Farmland\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Rectangular Farmland (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Rectangular Farmland training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'river': '[SPECTRAL SIGNATURE]\n- Water reflectance: Very low VIS (R~0.02-0.06) -- strong absorption\n- NIR: Near-zero (strong absorption) -- primary water discriminator\n- NDWI (McFeeters): Positive (>0.2) -- reliable water detection index\n- Turbidity: Elevated red/green ratio in sediment-laden water (TSS proxy)\n- Algal bloom: Green band anomaly (Chl-a indicator in eutrophic reaches)\n\n[SPATIAL METRICS]\n- Channel width: 5m-5km (scale-dependent)\n- Sinuosity index: 1.0 (straight channel) to >2.5 (meandering/anastomosing)\n- Braiding index: Multi-thread pattern indicates high bedload transport\n- Floodplain width: Detectable via NDWI expansion during flood events\n\n[CLASSIFICATION FINDINGS]\n- Class: River (Linear inland water body -- LCCS W1)\n- Key discriminators: Dark linear feature + riparian vegetation fringe + NDWI positive\n- Geomorphic signature: Sinuosity encodes discharge and sediment regime\n- Confusion risk: Road/shadow (similar geometry) -- differentiated by NDWI\n\n[ENVIRONMENTAL INDICATORS]\n- TSS (Suspended Sediment): Red/NIR band ratio proxy\n- Flood stage: NDWI expansion vs. baseline water mask\n- Riparian corridor NDVI: Target >0.4 for healthy buffer\n- Water quality anomalies: Algal bloom index, turbidity mapping\n\n[APPLICATIONS]\nFlood inundation mapping | Water quality monitoring (TSS, Chl-a) | Hydropower siting | Drought streamflow estimation | Aquatic habitat connectivity',
    'roundabout': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Roundabout\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Roundabout (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Roundabout training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'runway': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Runway\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Runway (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Runway training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'sea_ice': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Sea Ice\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Sea Ice (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Sea Ice training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'ship': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Ship\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Ship (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Ship training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'snowberg': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Snowberg\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Snowberg (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Snowberg training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'sparse_residential': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Sparse Residential\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Sparse Residential (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Sparse Residential training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'stadium': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Stadium\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Stadium (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Stadium training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'storage_tank': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Storage Tank\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Storage Tank (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Storage Tank training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'tennis_court': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Tennis Court\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Tennis Court (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Tennis Court training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'terrace': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Terrace\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Terrace (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Terrace training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'thermal_power_station': '[SPECTRAL SIGNATURE]\n- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n- NDVI: Dependent on vegetation fraction within classified extent\n- LST: Varies by surface material and land-use activity intensity\n\n[SPATIAL METRICS]\n- Texture: Characteristic spatial frequency and pattern for Thermal Power Station\n- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n[CLASSIFICATION FINDINGS]\n- Class: Thermal Power Station (LCCS-compliant land cover category)\n- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n- Training similarity: Spectral-spatial feature match to Thermal Power Station training tiles\n- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n[ENVIRONMENTAL INDICATORS]\n- ISC: Class-specific impervious surface cover estimate\n- NDVI/EVI range: Vegetation fraction and health indicator\n- LST anomaly: Thermal profile relative to rural reference surface\n\n[APPLICATIONS]\nLULC change detection | Urban planning support | Environmental impact assessment | Remote sensing classification benchmarking',
    'wetland': '[SPECTRAL SIGNATURE]\n- Saturated soil: Low NIR, moderate VIS -- darker than dry soil\n- Emergent vegetation: Moderate NDVI (0.3-0.6) -- distinct from upland forest\n- Open water patches: NDWI positive (>0.2)\n- Peat/organic soil: Very low reflectance across all bands\n- Spectral pattern: Heterogeneous mosaic (NDVI/NDWI hybrid) -- diagnostic\n\n[SPATIAL METRICS]\n- Surface heterogeneity: High (interspersed water, emergent veg, mud)\n- Boundary geometry: Diffuse/gradational (not sharp)\n- Texture: Mixed fine-coarse by vegetation density\n- Seasonal extent: NDWI time-series essential for full mapping\n\n[CLASSIFICATION FINDINGS]\n- Class: Wetland (LCCS W2 -- Inland non-permanent/permanent water)\n- Sub-types: Marsh | Swamp | Mangrove (coastal) | Peatland | Floodplain\n- Key discriminators: Hybrid NDVI/NDWI pattern + seasonal inundation signal\n- Confusion risk: Rice paddy (similar signal) -- SAR-C backscatter differentiates\n\n[ENVIRONMENTAL INDICATORS]\n- Blue Carbon: 3-5x forest carbon per unit area (mangroves, peatlands)\n- Biodiversity: >35% of all species depend on wetland habitats\n- Flood regulation: Peak flow attenuation capacity\n- CH4 emissions: Significant anaerobic decomposition source\n\n[APPLICATIONS]\nRamsar site monitoring | Blue carbon accounting | Flood storage mapping | Biodiversity conservation planning | Water purification service valuation',
}


# ---------------------------
# Database Connection
# ---------------------------
def get_db_connection():
    return mysql.connector.connect(
        host = os.environ.get("DB_HOST"),
        user = os.environ.get("DB_USER"),
        password = os.environ.get("DB_PASSWORD"), 
        database = os.environ.get("DB_NAME")
    )

# ---------------------------
# Cloud configuration
# ---------------------------

cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key = os.environ.get("CLOUDINARY_API_KEY"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
)

print(os.environ.get("CLOUDINARY_API_KEY"))

# ---------------------------
# HOME PAGE
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

# ---------------------------
# SIGN UP
# ---------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['uname']
        email = request.form['email']
        password = request.form['password']

        # Basic validation
        if not uname.strip():
            flash("Username is required", "danger")
            return redirect(url_for('register'))

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email address", "danger")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check existing email
        cursor.execute("SELECT u_id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash("Email already registered", "danger")
            cursor.close()
            conn.close()
            return redirect(url_for('register'))

        # Insert user
        cursor.execute(
            "INSERT INTO users (uname, email, password) VALUES (%s, %s, %s)",
            (uname, email, hashed_password)
        )
        conn.commit()

        cursor.close()
        conn.close()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# ---------------------------
# LOGIN
# ---------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email address", "danger")
            return redirect(url_for('login'))

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['u_id']
            session['username'] = user['uname']
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

# ---------------------------
# LOGOUT 
# ---------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------------------------
# UPLOAD
# ---------------------------

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if not session.get('user_id'):
        return redirect(url_for('login'))

    if request.method == 'POST':

        location = request.form.get('location')
        file = request.files.get('image')

        if not location:
            flash("Location is required", "danger")
            return redirect(url_for('upload'))

        if not file or file.filename == '':
            flash("Image file is required", "danger")
            return redirect(url_for('upload'))

        # -----------------------------
        # Save temporarily to local
        # -----------------------------
        filename = secure_filename(file.filename)
        local_path = os.path.join("static/uploads", filename)
        file.save(local_path)

        # -----------------------------
        # Prediction
        # -----------------------------
        img = image.load_img(local_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img)

        # --- CLAHE Preprocessing: normalises contrast/brightness for real-world images ---
        img_bgr = cv2.cvtColor(img_arr.astype('uint8'), cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        img_lab_eq = cv2.merge([l_eq, a, b])
        img_rgb_eq = cv2.cvtColor(cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        img_arr = img_rgb_eq.astype('float32') / 255.0
        # ---------------------------------------------------------------------------------

        img_arr = np.expand_dims(img_arr, axis=0)
        preds = model.predict(img_arr)
        
        # Get top 5 predictions
        top_indices = np.argsort(preds[0])[-5:][::-1]
        top_predictions = [
            {"class_name": CLASS_NAMES[i], "confidence": round(float(preds[0][i]) * 100, 2)}
            for i in top_indices
        ]
        
        class_idx = top_indices[0] # Best match
        confidence = top_predictions[0]["confidence"]
        predicted_class = CLASS_NAMES[class_idx]
        
        detailed_report = DETAILED_REPORTS.get(predicted_class, "No detailed analysis available for this classification.")

        # -----------------------------
        # Upload to Cloudinary
        # -----------------------------
        # Clean location name (remove spaces & lowercase)
        clean_location = location.strip().lower().replace(" ", "_")

        upload_result = cloudinary.uploader.upload(
            local_path,
            folder=f"geonex/{clean_location}"
        )

        cloud_url = upload_result["secure_url"]
        public_id = upload_result["public_id"]

        # -----------------------------
        # Delete local copy
        # -----------------------------
        os.remove(local_path)

        # -----------------------------
        # Insert into Database
        # -----------------------------
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO image_records
            (user_id, image_url, public_id, prediction, confidence, location, top_predictions, detailed_report)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session.get("user_id"),
            cloud_url,
            public_id,
            predicted_class,
            round(confidence, 2),
            location,
            json.dumps(top_predictions),
            detailed_report
        ))

        conn.commit()
        cursor.close()
        conn.close()

        # -----------------------------
        # Store result in session (PRG)
        # -----------------------------
        session['result'] = {
            "image": cloud_url,
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "location": location,
            "top_predictions": top_predictions,
            "detailed_report": detailed_report
        }

        return redirect(url_for('upload'))

    # -----------------------------
    # GET request
    # -----------------------------
    result = session.get('result', None)  # Use get (not pop) so /print-report can still read it

    if result:
        return render_template(
            "upload.html",
            uploaded_image=result["image"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            location=result["location"],
            top_predictions=result.get("top_predictions", []),
            detailed_report=result.get("detailed_report", "")
        )

    return render_template("upload.html")

# ---------------------------
# PRINT / VIEW REPORT
# ---------------------------
@app.route('/print-report')
def print_report():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    result = session.get('result')
    if not result:
        flash("No report available. Please upload an image first.", "danger")
        return redirect(url_for('upload'))
    return render_template(
        "print_report.html",
        uploaded_image=result.get("image", ""),
        prediction=result.get("prediction", ""),
        confidence=result.get("confidence", 0),
        location=result.get("location", ""),
        top_predictions=result.get("top_predictions", []),
        detailed_report=result.get("detailed_report", "")
    )

from datetime import date

@app.route('/report', methods=['GET', 'POST'])
def report():

    today = date.today().isoformat()  
    
    if not session.get('user_id'):
        return redirect(url_for('login')) 

    if request.method == 'POST':

        location = request.form.get('location')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        session['report_data'] = {
            "location": location,
            "start_date": start_date,
            "end_date": end_date
        }

        return redirect(url_for('report'))

    # ---------- GET ----------

    report_data = session.pop('report_data', None)

    if report_data:

        location = report_data['location']
        start_date = report_data['start_date']
        end_date = report_data['end_date']

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        cursor.execute("""
            SELECT prediction AS class_name,
                   COUNT(*) AS count
            FROM image_records
            WHERE location = %s
              AND DATE(uploaded_at) BETWEEN %s AND %s
            GROUP BY prediction
        """, (location, start_date, end_date))

        folders = cursor.fetchall()

        cursor.close()
        db.close()

        return render_template(
            "report.html",
            folders=folders,
            show_result_section=True,
            location=location,
            start_date=start_date,
            end_date=end_date,
            today=today
        )

    # Normal GET → empty
    return render_template(
        "report.html",
        folders=None,
        today=today,
        show_result_section=False
    )

@app.route('/report-images')
def report_images():

    selected_class = request.args.get('selected_class')
    location = request.args.get('location')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM image_records
        WHERE location = %s
          AND prediction = %s
          AND DATE(uploaded_at) BETWEEN %s AND %s
        ORDER BY uploaded_at DESC
    """, (location, selected_class, start_date, end_date))

    images = cursor.fetchall()

    for img in images:
        if img.get("top_predictions"):
            try:
                img["top_predictions_parsed"] = json.loads(img["top_predictions"])
            except:
                img["top_predictions_parsed"] = []
        else:
            img["top_predictions_parsed"] = []

    cursor.close()
    db.close()

    return render_template(
        "report_images.html",
        images=images,
        selected_class=selected_class,
        location=location,
        start_date=start_date,
        end_date=end_date
    )
# ---------------------------
# RUN APP
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, port=4000)