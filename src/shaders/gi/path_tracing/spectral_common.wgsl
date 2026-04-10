// spectral_common.wgsl
// Spectral rendering utilities — pbrt v4 hero-wavelength approach.
// Included AFTER pt_common.wgsl by all PT compute shaders.
//
// Reference: Physically Based Rendering v4 (Pharr, Jakob, Humphreys)
//   - Chapter 4: Radiometry, Spectra, and Color
//   - Hero wavelength sampling: Wilkie et al. 2014

// ============================================================
// Constants
// ============================================================
const LAMBDA_MIN: f32 = 360.0;  // nm
const LAMBDA_MAX: f32 = 830.0;  // nm
const LAMBDA_RANGE: f32 = 470.0; // LAMBDA_MAX - LAMBDA_MIN
const N_SPECTRAL_SAMPLES: u32 = 4u;

// Normalization constant for CIE Y integral (≈106.857)
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
// Spectrum → XYZ → sRGB
// ============================================================

// Convert 4 spectral samples to CIE XYZ tristimulus values.
// Uses Monte Carlo integration: XYZ = (1/N) * Σ S(λ) * [x̄/ȳ/z̄](λ) / pdf(λ)
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

    // Average over N samples, normalize by CIE Y integral
    xyz /= f32(N_SPECTRAL_SAMPLES);
    // The CIE functions are defined such that ∫ȳ(λ)dλ = CIE_Y_INTEGRAL
    // For a uniform wavelength distribution over [360, 830], the MC estimator gives:
    // X = LAMBDA_RANGE * (1/N) * Σ S(λi) * x̄(λi)
    // We want physical units, but for rendering we just need relative scaling.
    // With pdf = 1/LAMBDA_RANGE, the 1/pdf cancels with the normalization.
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
// RGB → Spectral Reflectance (Smits 1999, simplified)
// Converts an sRGB triplet to a smooth spectral reflectance
// evaluated at the given wavelengths.
// ============================================================

// Basis spectral functions: white, cyan, magenta, yellow, red, green, blue
// Smits uses piecewise-linear SPDs; we approximate with smooth Gaussian shapes.

fn smitsWhite(lambda: f32) -> f32 {
    // Approximately flat across the visible spectrum
    return 1.0;
}

fn smitsRed(lambda: f32) -> f32 {
    // Red component: peaks around 620nm, zero below ~580nm
    return smoothstep(580.0, 620.0, lambda);
}

fn smitsGreen(lambda: f32) -> f32 {
    // Green component: peaks around 540nm, Gaussian-like
    let t = (lambda - 540.0) / 60.0;
    return exp(-0.5 * t * t);
}

fn smitsBlue(lambda: f32) -> f32 {
    // Blue component: peaks around 460nm, fades above 490nm
    return 1.0 - smoothstep(460.0, 530.0, lambda);
}

fn smitsCyan(lambda: f32) -> f32 {
    // Cyan = green + blue region
    return select(1.0, smoothstep(420.0, 480.0, lambda), lambda < 480.0)
         * select(1.0, 1.0 - smoothstep(590.0, 660.0, lambda), lambda > 590.0);
}

fn smitsMagenta(lambda: f32) -> f32 {
    // Magenta = red + blue (dip in green)
    return max(smitsRed(lambda), smitsBlue(lambda))
         * (1.0 - 0.7 * smitsGreen(lambda));
}

fn smitsYellow(lambda: f32) -> f32 {
    // Yellow = red + green (no blue)
    return 1.0 - smitsBlue(lambda);
}

// Main RGB-to-Spectrum uplifting function.
// Uses Smits-style decomposition: RGB → white + primary + secondary
fn rgbToSpectrum(rgb: vec3f, lambdas: vec4f) -> vec4f {
    var result = vec4f(0.0);

    for (var i = 0u; i < N_SPECTRAL_SAMPLES; i++) {
        let lambda = lambdas[i];
        var r = rgb.r;
        var g = rgb.g;
        var b = rgb.b;

        var spd = 0.0;

        // Decompose: extract white component first (minimum of R,G,B)
        let white = min(r, min(g, b));
        r -= white;
        g -= white;
        b -= white;
        spd += white * smitsWhite(lambda);

        if (r > 0.0 && g > 0.0) {
            // Yellow region
            let yellow = min(r, g);
            r -= yellow;
            g -= yellow;
            spd += yellow * smitsYellow(lambda);
        } else if (r > 0.0 && b > 0.0) {
            // Magenta region
            let magenta = min(r, b);
            r -= magenta;
            b -= magenta;
            spd += magenta * smitsMagenta(lambda);
        } else if (g > 0.0 && b > 0.0) {
            // Cyan region
            let cyan = min(g, b);
            g -= cyan;
            b -= cyan;
            spd += cyan * smitsCyan(lambda);
        }

        // Remaining primaries
        spd += r * smitsRed(lambda);
        spd += g * smitsGreen(lambda);
        spd += b * smitsBlue(lambda);

        result[i] = max(spd, 0.0);
    }

    return result;
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
