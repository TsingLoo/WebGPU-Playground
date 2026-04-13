// spectral_common.wgsl
// Spectral rendering utilities — pbrt v4 hero-wavelength approach.
// Included AFTER pt_common.wgsl by all PT compute shaders.
//
// Reference: Physically Based Rendering v4 (Pharr, Jakob, Humphreys)
//   - Chapter 4: Radiometry, Spectra, and Color
//   - Hero wavelength sampling: Wilkie et al. 2014
//
// COLOR PIPELINE:
//   rgbToSpectrum()  — sRGB spectral response basis (guaranteed round-trip)
//   spectrumToRGB()  — direct spectral→sRGB with Gram matrix correction
//   Both use sRGB response functions = rows of XYZ→sRGB matrix × CIE CMFs.
//   The precomputed Gram matrix G' and its inverse G'^(-1) ensure that
//   E[ spectrumToRGB( rgbToSpectrum(c) ) ] = c  exactly.

// ============================================================
// Constants
// ============================================================
const LAMBDA_MIN: f32 = 360.0;  // nm
const LAMBDA_MAX: f32 = 830.0;  // nm
const LAMBDA_RANGE: f32 = 470.0; // LAMBDA_MAX - LAMBDA_MIN
const N_SPECTRAL_SAMPLES: u32 = 4u;

// CIE Y integral ∫ȳ(λ)dλ over [360,830] — used for physical XYZ normalization
const CIE_Y_INTEGRAL: f32 = 106.857;

// ============================================================
// SampledWavelengths: 4 stratified hero wavelengths + PDFs
// ============================================================

// Sample 4 stratified wavelengths using hero wavelength approach.
// u: uniform random in [0, 1)
// Returns vec4f of wavelengths in nm.
fn sampleHeroWavelengths(u: f32) -> vec4f {
    // Hero wavelength: uniform in [LAMBDA_MIN, LAMBDA_MAX)
    let lambda0 = LAMBDA_MIN + u * LAMBDA_RANGE;

    // 3 additional wavelengths at equal spacing (wrap around)
    let delta = LAMBDA_RANGE / 4.0;
    let lambda1 = LAMBDA_MIN + fract((lambda0 - LAMBDA_MIN + delta) / LAMBDA_RANGE) * LAMBDA_RANGE;
    let lambda2 = LAMBDA_MIN + fract((lambda0 - LAMBDA_MIN + 2.0 * delta) / LAMBDA_RANGE) * LAMBDA_RANGE;
    let lambda3 = LAMBDA_MIN + fract((lambda0 - LAMBDA_MIN + 3.0 * delta) / LAMBDA_RANGE) * LAMBDA_RANGE;

    return vec4f(lambda0, lambda1, lambda2, lambda3);
}

// PDF for uniformly sampled wavelengths
fn wavelengthPDF() -> vec4f {
    return vec4f(1.0 / LAMBDA_RANGE);
}

// ============================================================
// CIE 1931 Color Matching Functions (Wyman et al. 2013 Gaussian fit)
// Multi-lobe Gaussian approximation — accurate and GPU-friendly.
// ============================================================

fn gaussianTerm(lambda: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32 {
    let t = (lambda - mu);
    let sigma = select(sigma2, sigma1, t < 0.0);
    return exp(-0.5 * t * t / (sigma * sigma));
}

fn cie_x(lambda: f32) -> f32 {
    return 1.056 * gaussianTerm(lambda, 599.8, 37.9, 31.0)
         + 0.362 * gaussianTerm(lambda, 442.0, 16.0, 26.7)
         - 0.065 * gaussianTerm(lambda, 501.1, 20.4, 26.2);
}

fn cie_y(lambda: f32) -> f32 {
    return 0.821 * gaussianTerm(lambda, 568.8, 46.9, 40.5)
         + 0.286 * gaussianTerm(lambda, 530.9, 16.3, 31.1);
}

fn cie_z(lambda: f32) -> f32 {
    return 1.217 * gaussianTerm(lambda, 437.0, 11.8, 36.0)
         + 0.681 * gaussianTerm(lambda, 459.0, 26.0, 13.8);
}

// ============================================================
// sRGB Spectral Response Functions
// ============================================================
// resp_k(λ) = M_xyz2rgb[k,:] · [x̄(λ), ȳ(λ), z̄(λ)]
// These define the spectral sensitivity of each sRGB channel.

fn srgbResponse(lambda: f32) -> vec3f {
    let x = cie_x(lambda);
    let y = cie_y(lambda);
    let z = cie_z(lambda);
    return vec3f(
         3.2404542 * x - 1.5371385 * y - 0.4985314 * z,  // R
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,  // G
         0.0556434 * x - 0.2040259 * y + 1.0572252 * z   // B
    );
}

// ============================================================
// Gram Matrix Inverse for sRGB Spectral Round-Trip
// ============================================================
// G'[j][k] = ∫ resp_j(λ) * resp_k(λ) dλ  over [360, 830]
//
// Precomputed numerically using Wyman Gaussian CMFs:
// G' = [ 339.9437, -15.1686,  -4.8310 ]
//      [ -15.1686, 132.7275, -19.9267 ]
//      [  -4.8310, -19.9267, 158.5486 ]
//
// The inverse G'^(-1) corrects the MC estimator so that
// E[spectrumToRGB(rgbToSpectrum(c, λ), λ)] = c  exactly.
//
// Stored as column-major mat3x3f:
const GRAM_INV_00: f32 = 0.002960;
const GRAM_INV_01: f32 = 0.000359;
const GRAM_INV_02: f32 = 0.000135;
const GRAM_INV_10: f32 = 0.000359;
const GRAM_INV_11: f32 = 0.007723;
const GRAM_INV_12: f32 = 0.000982;
const GRAM_INV_20: f32 = 0.000135;
const GRAM_INV_21: f32 = 0.000982;
const GRAM_INV_22: f32 = 0.006435;

fn applyGramInverse(raw: vec3f) -> vec3f {
    return vec3f(
        GRAM_INV_00 * raw.x + GRAM_INV_01 * raw.y + GRAM_INV_02 * raw.z,
        GRAM_INV_10 * raw.x + GRAM_INV_11 * raw.y + GRAM_INV_12 * raw.z,
        GRAM_INV_20 * raw.x + GRAM_INV_21 * raw.y + GRAM_INV_22 * raw.z
    );
}

// ============================================================
// RGB → Spectral Reflectance (Bounded)
// ============================================================
// PBRT explicitly distinguishes between Reflectance and Illuminant spectra.
// For Reflectances (Albedo, F0), the response MUST be bounded within [0, 1] 
// to ensure energy conservation. The true Gram/CMF formulation produces 
// values >1.0 or <0.0 which causes exponential color explosions across bounces.
// We decompose RGB into an achromatic part (flat spectrum) and a chromatic part.

fn rgbToReflectanceSpectrum(rgb: vec3f, lambdas: vec4f) -> vec4f {
    let white = min(rgb.r, min(rgb.g, rgb.b));
    let color = rgb - vec3f(white);
    
    // Achromatic base reflects equally across all wavelengths safely
    var result = vec4f(white);
    
    for (var i = 0u; i < N_SPECTRAL_SAMPLES; i++) {
        let l = lambdas[i];
        
        // Distribute remaining chromatic energy safely via Gaussian curves
        let r_val = exp(-0.5 * ((l - 610.0) / 40.0) * ((l - 610.0) / 40.0));
        let g_val = exp(-0.5 * ((l - 540.0) / 40.0) * ((l - 540.0) / 40.0));
        let b_val = exp(-0.5 * ((l - 450.0) / 40.0) * ((l - 450.0) / 40.0));
        
        result[i] += color.r * r_val + color.g * g_val + color.b * b_val;
    }
    
    // Hard physics clamp
    return clamp(result, vec4f(0.0), vec4f(1.0));
}

// ============================================================
// RGB → Spectral Illuminant (Unbounded / Exact Round-Trip)
// ============================================================
// Uses sRGB spectral response basis functions for light sources.

fn rgbToIlluminantSpectrum(rgb: vec3f, lambdas: vec4f) -> vec4f {
    var result = vec4f(0.0);
    for (var i = 0u; i < N_SPECTRAL_SAMPLES; i++) {
        let resp = srgbResponse(lambdas[i]);
        result[i] = rgb.r * resp.x + rgb.g * resp.y + rgb.b * resp.z;
    }
    return result;
}
// ============================================================
// Spectrum → sRGB (Monte Carlo with Gram correction)
// ============================================================
// Direct spectral→RGB conversion, bypassing XYZ.
//
// MC estimator: raw[k] = (LAMBDA_RANGE/N) * Σ S(λi) * resp_k(λi)
// Then:         rgb = G'^(-1) * raw
//
// This guarantees E[spectrumToRGB(rgbToSpectrum(c))] = c exactly.

fn spectrumToRGB(spectrum: vec4f, lambdas: vec4f) -> vec3f {
    var raw = vec3f(0.0);
    for (var i = 0u; i < N_SPECTRAL_SAMPLES; i++) {
        let resp = srgbResponse(lambdas[i]);
        raw += spectrum[i] * resp;
    }
    // MC integration: multiply by LAMBDA_RANGE / N
    raw *= LAMBDA_RANGE / f32(N_SPECTRAL_SAMPLES);

    // Apply Gram matrix inverse for exact round-tripping
    return applyGramInverse(raw);
}

// ============================================================
// Spectrum → CIE XYZ (for physical quantities / emission)
// ============================================================
// Standard MC estimator with proper CIE normalization:
//   XYZ[j] = (1/(N*CIE_Y_INTEGRAL)) * Σ S(λi) * CMF_j(λi) / pdf(λi)

fn spectrumToXYZ(spectrum: vec4f, lambdas: vec4f, pdfs: vec4f) -> vec3f {
    var xyz = vec3f(0.0);
    for (var i = 0u; i < N_SPECTRAL_SAMPLES; i++) {
        let lambda = lambdas[i];
        let s = spectrum[i];
        let pdf = pdfs[i];

        if (pdf > 0.0) {
            let weight = s / pdf;
            xyz.x += weight * cie_x(lambda);
            xyz.y += weight * cie_y(lambda);
            xyz.z += weight * cie_z(lambda);
        }
    }
    xyz /= f32(N_SPECTRAL_SAMPLES) * CIE_Y_INTEGRAL;
    return xyz;
}

// CIE XYZ → linear sRGB (D65 white point, BT.709/sRGB primaries)
fn xyzToLinearSRGB(xyz: vec3f) -> vec3f {
    return vec3f(
         3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
        -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
         0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z
    );
}

// ============================================================
// Spectral Fresnel — wavelength-dependent IOR
// ============================================================

// Cauchy's equation for dispersion: n(λ) = A + B/λ² + C/λ⁴
// Simplified to 2 terms for performance.
// baseIOR is the IOR at reference wavelength ~550nm.
fn cauchyIOR(baseIOR: f32, lambda: f32) -> f32 {
    // Cauchy coefficients derived from baseIOR at 550nm
    // n(λ) ≈ A + B/λ²
    // At λ=550: baseIOR = A + B/(550²)
    // We set B so that n(450) - n(650) ≈ 0.03 (typical for glass)
    let B = 0.02 * 550.0 * 550.0; // dispersion strength
    let A = baseIOR - B / (550.0 * 550.0);
    return A + B / (lambda * lambda);
}

// Compute Fresnel reflectance for 4 wavelength samples
fn fresnelDielectricSpectral(cosI: f32, etaI: vec4f, etaT: vec4f) -> vec4f {
    var result = vec4f(0.0);
    for (var i = 0u; i < N_SPECTRAL_SAMPLES; i++) {
        result[i] = fresnelDielectric(cosI, etaI[i], etaT[i]);
    }
    return result;
}

// Compute 4 wavelength-dependent IOR values from base IOR
fn spectralIOR(baseIOR: f32, lambdas: vec4f) -> vec4f {
    return vec4f(
        cauchyIOR(baseIOR, lambdas[0]),
        cauchyIOR(baseIOR, lambdas[1]),
        cauchyIOR(baseIOR, lambdas[2]),
        cauchyIOR(baseIOR, lambdas[3])
    );
}

// ============================================================
// Spectral Fresnel-Schlick (for metals/conductors)
// ============================================================

// vec4f version: F0 per-wavelength
fn fresnelSchlickSpectral(cosTheta: f32, F0: vec4f) -> vec4f {
    let p = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (vec4f(1.0) - F0) * p;
}
