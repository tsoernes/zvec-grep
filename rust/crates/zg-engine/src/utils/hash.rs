use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

use sha2::{Digest, Sha256};

pub(crate) fn sha256_parts<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> [u8; 32] {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update(part);
    }
    digest.finalize().into()
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    sha256_hex_parts([bytes])
}

pub(crate) fn sha256_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 64 * 1024].into_boxed_slice();
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().into())
}

pub(crate) fn sha256_hex_parts<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    hex::encode(sha256_parts(parts))
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn matches_known_sha256_digests() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
