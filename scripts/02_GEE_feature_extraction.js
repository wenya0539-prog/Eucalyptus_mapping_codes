// ==============================================================================
// Landsat time-series feature extraction for Eucalyptus plantation mapping.
// Landsat 5/7 surface reflectance is harmonized to the Landsat 8 reference system.
// ==============================================================================

// ==================== 1. 核心参数设置 ====================
var TARGET_YEAR = 2020;        // 目标年份（本数据集为 2001–2020）
var SENSOR_MODE = 'AUTO';      // 'AUTO'为自动选择最佳卫星，也可强制指定 'L5', 'L7', 'L8'
var START_MONTH = 1;           // 开始月份 
var END_MONTH = 12;            // 结束月份
var PROVINCE_NAME = 'Zhejiang'; // 与 GEE 矢量资产对应的省份英文名
var VECTOR_PATH = 'projects/clean-yew-443009-r2/assets/zhejiang'; // GEE 中的省级边界资产 ID
var EXPORT_NAME = PROVINCE_NAME + '_' + TARGET_YEAR;

// ==================== 2. 传感器智能路由逻辑 ====================
var sensorType;
if (SENSOR_MODE === 'AUTO') {
  if (TARGET_YEAR >= 2013) { sensorType = 'L8'; }
  else if (TARGET_YEAR === 2012) { sensorType = 'L7'; }
  else { sensorType = 'L5'; }
} else {
  sensorType = SENSOR_MODE;
}
print('🎯 目标年份: ' + TARGET_YEAR, '🛰️ 启用传感器: ' + sensorType);

// ==================== 3. 初始化与基础设置 ====================
var table = ee.FeatureCollection(VECTOR_PATH).geometry();
Map.centerObject(table, 10);
var startDate = ee.Date.fromYMD(TARGET_YEAR, START_MONTH, 1);
var endDate = ee.Date.fromYMD(TARGET_YEAR, END_MONTH, 31).advance(1, 'day');

// ==================== 4. 传感器专有预处理函数 ====================

// --- 4.1 Landsat 8/9 预处理 ---
function prepL8(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0))
               .and(qa.bitwiseAnd(1 << 4).eq(0)).and(qa.bitwiseAnd(1 << 5).eq(0));
               
  var scaled = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                    .multiply(0.0000275).add(-0.2)
                    .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
                    
  return scaled.updateMask(mask).copyProperties(image, image.propertyNames());
}

// --- 4.2 Landsat 5/7 预处理 (含 Roy 2016 统合至 L8 标准) ---
function prepL57(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0))
               .and(qa.bitwiseAnd(1 << 4).eq(0));
               
  var scaled = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
                    .multiply(0.0000275).add(-0.2)
                    .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
  
  // ★ 核心：Roy et al. (2016) 辐射统合 (L5/7 -> L8)
  var slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]);
  var intercepts = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]);
  var harmonized = scaled.multiply(slopes).add(intercepts);
  
  return harmonized.updateMask(mask).copyProperties(image, image.propertyNames());
}

// ==================== 5. 获取并统合影像 ====================
var baseCollection;
if (sensorType === 'L8') {
  baseCollection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(startDate, endDate).filterBounds(table).map(prepL8);
} else if (sensorType === 'L7') {
  baseCollection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate(startDate, endDate).filterBounds(table).map(prepL57);
} else if (sensorType === 'L5') {
  baseCollection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterDate(startDate, endDate).filterBounds(table).map(prepL57);
}

// 提取中值并确保数值合法 (限制在0-1之间，防止指数计算溢出报错)
var unifiedImage = baseCollection.median().clip(table).clamp(0, 1);
print('✨ 统合后的基准反射率影像:', unifiedImage);
Map.addLayer(unifiedImage, {bands: ['red', 'green', 'blue'], min: 0, max: 0.15}, TARGET_YEAR + ' 真彩色 (L8标准)', false);

// ==================== 6. 特征提取模块 (全适配 L8 标准) ====================

// --- 模块 A: 植被指数 (12个 - 经严格学术核对版) ---
function addVIs(img) {
  var eps = 1e-9; // 极小值，避免除0报错
  var ndvi = img.normalizedDifference(['nir', 'red']).rename('NDVI');
  var nbr = img.normalizedDifference(['nir', 'swir2']).rename('NBR'); // 修正：使用 SWIR2
  var mndwi = img.normalizedDifference(['green', 'swir1']).rename('MNDWI');
  var rgri = img.select('red').divide(img.select('green').add(eps)).clamp(-5, 5).rename('RGRI');
  var evi = img.expression('2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)', 
            {nir: img.select('nir'), red: img.select('red'), blue: img.select('blue'), eps: eps}).clamp(-1, 1).rename('EVI');
  var gci = img.select('nir').divide(img.select('green').add(eps)).subtract(1).clamp(-5, 10).rename('GCI');
  var sipi = img.expression('(nir - blue) / (nir - red + eps)', 
             {nir: img.select('nir'), blue: img.select('blue'), red: img.select('red'), eps: eps}).clamp(-1, 3).rename('SIPI');
  var vari = img.expression('(green - red) / (green + red - blue + eps)', 
             {green: img.select('green'), red: img.select('red'), blue: img.select('blue'), eps: eps}).clamp(-1, 1).rename('VARI');
  var ndgi = img.normalizedDifference(['green', 'red']).rename('NDGI');
var nrgdi = img.normalizedDifference(['red', 'green']).rename('NRGDI');
  var gbndvi = img.expression('(nir - green - blue) / (nir + green + blue + eps)', 
               {nir: img.select('nir'), green: img.select('green'), blue: img.select('blue'), eps: eps}).clamp(-1, 1).rename('GBNDVI');
  var bndvi = img.normalizedDifference(['nir', 'blue']).rename('BNDVI');
  
  return img.addBands([ndvi, nbr, mndwi, rgri, evi, gci, sipi, vari, ndgi, nrgdi, gbndvi, bndvi]);
}

// --- 模块 B: GLCM 纹理 (6个) ---
function addGLCM(img) {
  // 将反射率映射到 0-10000 整数用于 GLCM 计算
  var imgInt = img.select(['blue', 'green', 'red']).multiply(10000).int16().max(0);
  var glcm = imgInt.glcmTexture({size: 3, average: true});
  
  var glcmBands = ee.Image([
    glcm.select('blue_contrast'), glcm.select('blue_diss'),
    glcm.select('green_contrast'), glcm.select('green_diss'),
    glcm.select('red_contrast'), glcm.select('red_diss')
  ]);
  return img.addBands(glcmBands);
}

// --- 模块 C: 缨帽变换 (5个) - 强制使用 L8 Baig 2014 系数 ---
function addTasseledCap(img) {
  var tcInput = img.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
  var coeffs = ee.Array([
    [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],
    [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],
    [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]
  ]);
  var tc = ee.Image(coeffs).matrixMultiply(tcInput.toArray().toArray(1))
                           .arrayProject([0])
                           .arrayFlatten([['brightness', 'greenness', 'wetness']]);
  var tcd = tc.select('brightness').pow(2).add(tc.select('greenness').pow(2)).sqrt().rename('TCD');
  var tca = tc.select('greenness').divide(tc.select('brightness')).atan().rename('TCA');
  return img.addBands([tc, tcd, tca]);
}

// --- 模块 D: 地形特征 (3个) ---
function addTopo(img) {
  var dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation');
  var terrain = ee.Algorithms.Terrain(dem);
  var topoBands = terrain.select(['elevation', 'slope', 'aspect']);
  return img.addBands(topoBands);
}

// ==================== 7. 执行计算与图层显示 ====================
print('=== 开始叠加特征 ===');
var finalFeatures = addVIs(unifiedImage);
finalFeatures = addGLCM(finalFeatures);
finalFeatures = addTasseledCap(finalFeatures);
finalFeatures = addTopo(finalFeatures).float();

var allBands = finalFeatures.bandNames().getInfo();
print('提取特征数量: ' + allBands.length, allBands);

// 为了避免原图与统合图色差过大，这里的显示做了标准化拉伸
Map.addLayer(finalFeatures.select('NDVI'), {min: 0, max: 0.8, palette: ['white', 'green']}, TARGET_YEAR + ' NDVI', true);

// ==================== 8. 导出数据到 Google Drive (已激活) ====================
print('导出任务已生成，请在 Tasks 面板中运行。');

Export.image.toDrive({
  image: finalFeatures,              // 包含反射率 + 32维特征的最终影像
  description: EXPORT_NAME,          // 任务描述
  folder: 'GEE_Eucalyptus_Export',   // 云盘文件夹名称 (将自动创建)
  region: table,                     // 研究区边界
  scale: 30,                         // 30米空间分辨率
  crs: 'EPSG:32650',                 // 投影带 (WGS84 UTM 50N)
  maxPixels: 1e13                    // 防止大图报错
});