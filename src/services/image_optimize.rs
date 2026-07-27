//! Best-effort size and dimension normalization for image attachments.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

const MAX_EDGE_PX: u32 = 2000;

pub(crate) const BYTE_BUDGET: usize = 512_000;

/// Limits allocation from malicious or malformed image headers.
const MAX_DECODE_PIXELS: u64 = 40_000_000;

const MAX_REFINE_STEPS: u32 = 3;

pub struct OptimizedImage {
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

#[derive(Clone, Copy)]
enum SniffedFormat {
    Png,
    Jpeg,
}

/// Returns `None` when the original should be retained.
pub fn optimize_image(bytes: &[u8]) -> Option<OptimizedImage> {
    let format = sniff_format(bytes)?;
    let (width, height) = probe_dimensions(bytes, format)?;
    let over_dims = width > MAX_EDGE_PX || height > MAX_EDGE_PX;
    if bytes.len() <= BYTE_BUDGET && !over_dims {
        return None;
    }
    if u64::from(width) * u64::from(height) > MAX_DECODE_PIXELS {
        return None;
    }
    let rgb = decode_rgb(bytes, format, width, height)?;
    let (rgb, out_w, out_h) = match clamped_dimensions(width, height) {
        Some((nw, nh)) => (box_downscale_rgb(&rgb, width, height, nw, nh), nw, nh),
        None => (rgb, width, height),
    };
    let encoded = encode_near_budget(&rgb, out_w, out_h)?;
    if encoded.len() >= bytes.len() && !over_dims {
        return None;
    }
    Some(OptimizedImage {
        bytes: encoded,
        mime_type: "image/jpeg".to_string(),
    })
}

/// Optimizes a base64 payload and returns its rewritten MIME type and data.
pub fn optimize_base64(data: &str) -> Option<(String, String)> {
    let bytes = BASE64.decode(data).ok()?;
    let image = optimize_image(&bytes)?;
    Some((image.mime_type, BASE64.encode(image.bytes)))
}

fn sniff_format(bytes: &[u8]) -> Option<SniffedFormat> {
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => Some(SniffedFormat::Png),
        [0xFF, 0xD8, 0xFF, ..] => Some(SniffedFormat::Jpeg),
        _ => None,
    }
}

/// Reads dimensions before allocating a decoded pixel buffer.
fn probe_dimensions(bytes: &[u8], format: SniffedFormat) -> Option<(u32, u32)> {
    let (w, h) = match format {
        SniffedFormat::Png => {
            let mut decoder = zune_png::PngDecoder::new(bytes);
            decoder.decode_headers().ok()?;
            decoder.get_dimensions()?
        }
        SniffedFormat::Jpeg => {
            let mut decoder = zune_jpeg::JpegDecoder::new(bytes);
            decoder.decode_headers().ok()?;
            decoder.dimensions()?
        }
    };
    Some((u32::try_from(w).ok()?, u32::try_from(h).ok()?))
}

/// Flattens alpha onto white to keep transparent screenshots readable as JPEG.
fn decode_rgb(bytes: &[u8], format: SniffedFormat, width: u32, height: u32) -> Option<Vec<u8>> {
    let rgb = match format {
        SniffedFormat::Png => decode_png_rgb(bytes)?,
        SniffedFormat::Jpeg => decode_jpeg_rgb(bytes)?,
    };
    (rgb.len() == width as usize * height as usize * 3).then_some(rgb)
}

fn decode_png_rgb(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = zune_png::PngDecoder::new(bytes);
    let buf = match decoder.decode().ok()? {
        zune_core::result::DecodingResult::U8(buf) => buf,
        // 16-bit: the high byte is the standard 16→8 reduction.
        zune_core::result::DecodingResult::U16(buf) => {
            buf.into_iter().map(|v| (v >> 8) as u8).collect()
        }
        _ => return None,
    };
    let flatten = |value: u8, alpha: u8| -> u8 {
        ((u16::from(value) * u16::from(alpha) + 255 * (255 - u16::from(alpha))) / 255) as u8
    };
    // Palette images decode as RGB/RGBA (tRNS promotes to the alpha variant).
    match decoder.get_colorspace()? {
        zune_core::colorspace::ColorSpace::RGB => Some(buf),
        zune_core::colorspace::ColorSpace::RGBA => Some(
            buf.chunks_exact(4)
                .flat_map(|p| {
                    [
                        flatten(p[0], p[3]),
                        flatten(p[1], p[3]),
                        flatten(p[2], p[3]),
                    ]
                })
                .collect(),
        ),
        zune_core::colorspace::ColorSpace::Luma => {
            Some(buf.iter().flat_map(|&g| [g, g, g]).collect())
        }
        zune_core::colorspace::ColorSpace::LumaA => Some(
            buf.chunks_exact(2)
                .flat_map(|p| {
                    let g = flatten(p[0], p[1]);
                    [g, g, g]
                })
                .collect(),
        ),
        _ => None,
    }
}

fn decode_jpeg_rgb(bytes: &[u8]) -> Option<Vec<u8>> {
    zune_jpeg::JpegDecoder::new(bytes).decode().ok()
}

fn clamped_dimensions(width: u32, height: u32) -> Option<(u32, u32)> {
    let long_edge = width.max(height);
    if long_edge <= MAX_EDGE_PX {
        return None;
    }
    let scale = |edge: u32| -> u32 {
        let scaled = u64::from(edge) * u64::from(MAX_EDGE_PX) / u64::from(long_edge);
        u32::try_from(scaled).unwrap_or(MAX_EDGE_PX).max(1)
    };
    Some((scale(width), scale(height)))
}

fn box_downscale_rgb(src: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let span = |i: u32, n: u32, edge: u32| -> (usize, usize) {
        let lo = (u64::from(i) * u64::from(edge) / u64::from(n)) as usize;
        let hi = ((u64::from(i + 1) * u64::from(edge) / u64::from(n)) as usize).max(lo + 1);
        (lo, hi)
    };
    let x_spans: Vec<(usize, usize)> = (0..nw).map(|x| span(x, nw, w)).collect();
    let (w, nw) = (w as usize, nw as usize);
    let mut out = vec![0u8; nw * nh as usize * 3];
    for y in 0..nh {
        let (sy0, sy1) = span(y, nh, h);
        for (x, &(sx0, sx1)) in x_spans.iter().enumerate() {
            let mut acc = [0u64; 3];
            for sy in sy0..sy1 {
                let row = &src[(sy * w + sx0) * 3..(sy * w + sx1) * 3];
                for px in row.chunks_exact(3) {
                    acc[0] += u64::from(px[0]);
                    acc[1] += u64::from(px[1]);
                    acc[2] += u64::from(px[2]);
                }
            }
            let count = ((sy1 - sy0) * (sx1 - sx0)) as u64;
            let dst = (y as usize * nw + x) * 3;
            out[dst] = (acc[0] / count) as u8;
            out[dst + 1] = (acc[1] / count) as u8;
            out[dst + 2] = (acc[2] / count) as u8;
        }
    }
    out
}

fn encode_near_budget(rgb: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let first = encode_jpeg(rgb, width, height, 90)?;
    if first.len() <= BYTE_BUDGET {
        return Some(first);
    }
    let mut quality = (90 * BYTE_BUDGET / first.len()).clamp(1, 89) as u8;
    let (mut lo, mut hi) = (1u8, 89u8);
    let mut fit: Option<Vec<u8>> = None;
    let mut smallest = first;
    for _ in 0..MAX_REFINE_STEPS {
        let candidate = encode_jpeg(rgb, width, height, quality)?;
        if candidate.len() <= BYTE_BUDGET {
            fit = Some(candidate);
            lo = quality + 1;
        } else {
            if candidate.len() < smallest.len() {
                smallest = candidate;
            }
            hi = quality.saturating_sub(1);
            if hi == 0 {
                break;
            }
        }
        if lo > hi {
            break;
        }
        quality = lo + (hi - lo) / 2;
    }
    Some(fit.unwrap_or(smallest))
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut out, quality);
    encoder
        .encode(
            rgb,
            u16::try_from(width).ok()?,
            u16::try_from(height).ok()?,
            jpeg_encoder::ColorType::Rgb,
        )
        .ok()?;
    Some(out)
}

#[cfg(test)]
pub(crate) fn test_png(w: u32, h: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&pixel(x, y));
            }
        }
        writer.write_image_data(&data).unwrap();
    }
    out
}

#[cfg(test)]
pub(crate) fn test_noise_png(w: u32, h: u32) -> Vec<u8> {
    test_png(w, h, test_noise_pixel)
}

#[cfg(test)]
pub(crate) fn test_noise_pixel(x: u32, y: u32) -> [u8; 4] {
    let mut v = u64::from(x)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(u64::from(y))
        .wrapping_mul(1442695040888963407);
    v ^= v >> 33;
    [v as u8, (v >> 8) as u8, (v >> 16) as u8, 255]
}

#[cfg(test)]
pub(crate) fn test_tiny_png() -> Vec<u8> {
    test_png(1, 1, |_, _| [9, 9, 9, 255])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded_dims(jpeg: &[u8]) -> (usize, usize) {
        let mut decoder = zune_jpeg::JpegDecoder::new(jpeg);
        decoder.decode_headers().unwrap();
        decoder.dimensions().unwrap()
    }

    #[test]
    fn small_image_passes_through_untouched() {
        let png = test_png(100, 80, |_, _| [200, 30, 30, 255]);
        assert!(optimize_image(&png).is_none());
    }

    #[test]
    fn oversized_dimensions_clamp_to_max_edge() {
        let png = test_png(3000, 1500, |x, _| [(x % 256) as u8, 120, 60, 255]);
        let out = optimize_image(&png).unwrap();
        assert_eq!(out.mime_type, "image/jpeg");
        assert_eq!(decoded_dims(&out.bytes), (2000, 1000));
    }

    #[test]
    fn over_budget_reencodes_smaller_as_jpeg() {
        let png = test_noise_png(1400, 1400);
        assert!(png.len() > BYTE_BUDGET);
        let out = optimize_image(&png).unwrap();
        assert_eq!(out.mime_type, "image/jpeg");
        assert!(out.bytes.len() < png.len());
        assert_eq!(decoded_dims(&out.bytes), (1400, 1400));
    }

    #[test]
    fn oversized_jpeg_input_shrinks() {
        let (w, h) = (1200u32, 1200u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&test_noise_pixel(x, y)[..3]);
            }
        }
        let jpeg = encode_jpeg(&rgb, w, h, 100).unwrap();
        assert!(jpeg.len() > BYTE_BUDGET);
        let out = optimize_image(&jpeg).unwrap();
        assert_eq!(out.mime_type, "image/jpeg");
        assert!(out.bytes.len() < jpeg.len());
    }

    #[test]
    fn transparency_flattens_onto_white() {
        let png = test_png(2100, 60, |_, _| [0, 0, 0, 0]);
        let out = optimize_image(&png).unwrap();
        assert_eq!(out.mime_type, "image/jpeg");
        let pixels = zune_jpeg::JpegDecoder::new(&out.bytes).decode().unwrap();
        assert!(pixels[..3].iter().all(|&c| c >= 250), "expected near-white");
    }

    #[test]
    fn undecodable_data_passes_through() {
        let mut fake = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        fake.extend(std::iter::repeat_n(0xAB, BYTE_BUDGET + 16));
        assert!(optimize_image(&fake).is_none());
    }

    #[test]
    fn unsupported_format_passes_through() {
        let mut gif = b"GIF89a".to_vec();
        gif.extend(std::iter::repeat_n(0xCD, BYTE_BUDGET + 16));
        assert!(optimize_image(&gif).is_none());
    }

    #[test]
    fn decode_ceiling_passes_bomb_through() {
        let mut out = Vec::new();
        let encoder = png::Encoder::new(&mut out, 8000, 6000);
        drop(encoder.write_header().unwrap());
        assert!(optimize_image(&out).is_none());
    }

    #[test]
    fn clamp_math_preserves_aspect_and_never_upscales() {
        assert_eq!(clamped_dimensions(1999, 1999), None);
        assert_eq!(clamped_dimensions(4000, 2000), Some((2000, 1000)));
        assert_eq!(clamped_dimensions(2000, 4000), Some((1000, 2000)));
        assert_eq!(clamped_dimensions(100_000, 10), Some((2000, 1)));
    }
}
