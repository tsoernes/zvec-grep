//! Validated V1 packed-sign records for the experimental native boundary.
//!
//! This codec does **not** enable or query a native `DiskANN` index. Each record
//! has least-significant-bit-first signs, zero-padded to a 32-bit word, then an
//! IEEE-754 scale encoded as four little-endian bytes. The logical dimension
//! is supplied separately and must match the eventual versioned index schema.

/// Maximum logical dimension supported by the proposed native V1 record.
pub const MAX_DIMENSIONS: usize = 20_000;

/// A malformed record or query was rejected before crossing the native ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationError {
    /// Dimensions must be between one and [`MAX_DIMENSIONS`].
    Dimension,
    /// The input does not have the required exact length.
    Length,
    /// An unused sign bit or alignment byte is nonzero.
    Padding,
    /// The document scale is negative or non-finite.
    Scale,
    /// A floating-point query component is non-finite.
    NonFiniteQuery,
    /// A zero query was rejected by the caller's metric policy.
    ZeroQuery,
}

/// Borrowed decoded signs and the corresponding document scale.
#[derive(Debug, PartialEq)]
pub struct RecordView<'a> {
    /// Packed signs without the 32-bit alignment padding.
    pub signs: &'a [u8],
    /// Per-document scale. Negative zero is preserved.
    pub scale: f32,
}

fn sizes(dimension: usize) -> Result<(usize, usize, usize), ValidationError> {
    if !(1..=MAX_DIMENSIONS).contains(&dimension) {
        return Err(ValidationError::Dimension);
    }
    let sign_bytes = dimension.div_ceil(8);
    let padded_bytes = dimension.div_ceil(32) * 4;
    Ok((sign_bytes, padded_bytes, padded_bytes + 4))
}

fn validate_signs(signs: &[u8], dimension: usize) -> Result<(), ValidationError> {
    let (sign_bytes, _, _) = sizes(dimension)?;
    if signs.len() != sign_bytes {
        return Err(ValidationError::Length);
    }
    let bits = dimension % 8;
    if bits != 0 && signs[sign_bytes - 1] & !((1u8 << bits) - 1) != 0 {
        return Err(ValidationError::Padding);
    }
    Ok(())
}

/// Encode signs and scale as a canonical V1 record.
///
/// # Errors
/// Returns an error for an invalid dimension, size, padding, or scale.
pub fn encode(signs: &[u8], scale: f32, dimension: usize) -> Result<Vec<u8>, ValidationError> {
    let (_, padded_bytes, record_bytes) = sizes(dimension)?;
    validate_signs(signs, dimension)?;
    if !scale.is_finite() || scale < 0.0 {
        return Err(ValidationError::Scale);
    }
    let mut record = vec![0; record_bytes];
    record[..signs.len()].copy_from_slice(signs);
    record[padded_bytes..].copy_from_slice(&scale.to_bits().to_le_bytes());
    Ok(record)
}

/// Decode and validate a canonical V1 record without copying its signs.
///
/// # Errors
/// Returns an error for an invalid dimension, size, padding, or scale.
pub fn decode(record: &[u8], dimension: usize) -> Result<RecordView<'_>, ValidationError> {
    let (sign_bytes, padded_bytes, record_bytes) = sizes(dimension)?;
    if record.len() != record_bytes {
        return Err(ValidationError::Length);
    }
    let signs = &record[..sign_bytes];
    validate_signs(signs, dimension)?;
    if record[sign_bytes..padded_bytes]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(ValidationError::Padding);
    }
    let scale = f32::from_bits(u32::from_le_bytes([
        record[padded_bytes],
        record[padded_bytes + 1],
        record[padded_bytes + 2],
        record[padded_bytes + 3],
    ]));
    if !scale.is_finite() || scale < 0.0 {
        return Err(ValidationError::Scale);
    }
    Ok(RecordView { signs, scale })
}

/// Validate a full-precision query without binarizing or modifying it.
///
/// # Errors
/// Returns an error for a bad dimension, length, non-finite value, or a zero
/// query when `reject_zero` is set (e.g. for cosine search).
pub fn validate_query(
    query: &[f32],
    dimension: usize,
    reject_zero: bool,
) -> Result<(), ValidationError> {
    sizes(dimension)?;
    if query.len() != dimension {
        return Err(ValidationError::Length);
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError::NonFiniteQuery);
    }
    if reject_zero && query.iter().all(|value| *value == 0.0) {
        return Err(ValidationError::ZeroQuery);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_DIMENSIONS, ValidationError, decode, encode, validate_query};

    #[test]
    fn canonical_v1_matches_native_little_endian_layout() {
        let signs = [0x15, 0x01];
        let record = encode(&signs, -0.0, 9).expect("encode V1 record");
        assert_eq!(record, [0x15, 0x01, 0, 0, 0, 0, 0, 0x80]);
        let view = decode(&record, 9).expect("decode V1 record");
        assert_eq!(view.signs, signs);
        assert_eq!(view.scale.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn roundtrips_non_word_dimensions_and_scale_bits() {
        for dim in [1, 7, 8, 9, 31, 32, 33, 65, 1024, MAX_DIMENSIONS] {
            let mut signs = vec![0xaa; dim.div_ceil(8)];
            if dim % 8 != 0 {
                *signs.last_mut().expect("nonempty") &= (1u8 << (dim % 8)) - 1;
            }
            for scale in [0.0, -0.0, 0.125, 1.5, 37.0] {
                let record = encode(&signs, scale, dim).expect("encode");
                let view = decode(&record, dim).expect("decode");
                assert_eq!(view.signs, signs);
                assert_eq!(view.scale.to_bits(), scale.to_bits());
            }
        }
    }

    #[test]
    fn rejects_bad_lengths_bits_padding_and_scales() {
        assert_eq!(encode(&[], 1.0, 0), Err(ValidationError::Dimension));
        assert_eq!(
            encode(&[0], 1.0, MAX_DIMENSIONS + 1),
            Err(ValidationError::Dimension)
        );
        assert_eq!(encode(&[], 1.0, 1), Err(ValidationError::Length));
        assert_eq!(encode(&[0x80], 1.0, 1), Err(ValidationError::Padding));
        for scale in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(encode(&[0], scale, 1), Err(ValidationError::Scale));
        }
        let valid = encode(&[1], 1.0, 1).expect("valid record");
        assert_eq!(decode(&valid[..7], 1), Err(ValidationError::Length));
        let mut bad = valid.clone();
        bad[1] = 1;
        assert_eq!(decode(&bad, 1), Err(ValidationError::Padding));
        bad = valid;
        bad[4..8].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert_eq!(decode(&bad, 1), Err(ValidationError::Scale));
    }

    #[test]
    fn validates_original_floating_query_without_quantization() {
        let query = [3.25, -0.125, 0.0];
        assert_eq!(validate_query(&query, 3, true), Ok(()));
        assert_eq!(
            query.map(f32::to_bits),
            [
                3.25_f32.to_bits(),
                (-0.125_f32).to_bits(),
                0.0_f32.to_bits()
            ]
        );
        assert_eq!(
            validate_query(&query, 2, false),
            Err(ValidationError::Length)
        );
        assert_eq!(
            validate_query(&[0.0, -0.0], 2, true),
            Err(ValidationError::ZeroQuery)
        );
        assert_eq!(validate_query(&[0.0, -0.0], 2, false), Ok(()));
        assert_eq!(
            validate_query(&[f32::INFINITY], 1, false),
            Err(ValidationError::NonFiniteQuery)
        );
    }
}
