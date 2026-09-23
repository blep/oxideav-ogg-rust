//! Ogg Media (OGM) identification-packet parsing.
//!
//! OGM predates the Xiph codec mappings: a logical stream is identified
//! by a one-byte `0x01` header flag followed by a five-character stream
//! class (`"video"`, `"audio"`, `"text"`). [`codec_id::detect`] matches
//! the Xiph signatures (`0x01 "vorbis"`, `0x80 "theora"`, …), so it
//! reports OGM streams as `unknown`; without this module the demuxer
//! would expose them as [`MediaType::Unknown`] with a 1 µs time base.
//! Real-world `.ogm` files carry MPEG-4 (Xvid/DivX) video and OGM text
//! subtitle streams, so classifying them matters for consumers.
//!
//! Layout, per the OGM container (Michael Ahlberg / Måns Rullgård,
//! `oggparseogm.c`): after `0x01` and the 5-byte class there is an
//! 8-byte prefix, then a class-specific tag — a little-endian FourCC
//! for video, a 4-byte ASCII-hex WAVE `wFormatTag` for audio. A common
//! tail follows at byte 13: LE32 `size`, LE64 `time_unit`, LE64 `spu`,
//! LE32 `default_len`, LE32 `buffersize`, LE32 `bits_per_sample`, then
//! video `width`/`height` (LE32 each) or audio `channels` (LE16) +
//! `block_align` (LE16) + `bit_rate` (LE32). One granule tick is
//! `time_unit / (spu * 10_000_000)` seconds — the time base ffmpeg
//! installs with
//! `avpriv_set_pts_info(st, 64, time_unit, spu * 10000000)`.
//!
//! [`codec_id::detect`]: crate::codec_id::detect
//! [`MediaType::Unknown`]: oxideav_core::MediaType::Unknown

use oxideav_core::{CodecId, CodecParameters, CodecTag, MediaType, TimeBase};

/// A parsed OGM identification packet.
#[derive(Clone, Debug)]
pub struct StreamHeader {
    /// Video, audio, or subtitle.
    pub media_type: MediaType,
    /// Codec id derived from the class-specific tag (e.g. `mpeg4` for an
    /// `XVID`/`DIVX` FourCC), or `text` for subtitle streams.
    pub codec_id: CodecId,
    /// On-wire tag: FourCC for video, `wFormatTag` for audio.
    pub tag: Option<CodecTag>,
    /// Video: coded width in pixels.
    pub width: Option<u32>,
    /// Video: coded height in pixels.
    pub height: Option<u32>,
    /// Audio: channel count.
    pub channels: Option<u16>,
    /// Audio: sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Timing numerator (`time_unit`), part of the granule time base.
    time_unit: u64,
    /// Timing denominator factor (`spu`), part of the granule time base.
    spu: u64,
}

impl StreamHeader {
    /// Parse an OGM identification packet, or `None` when `first` is not
    /// an OGM header (a Xiph mapping, a truncated packet, or unrelated
    /// bytes).
    pub fn parse(first: &[u8]) -> Option<Self> {
        if first.first().copied()? != 0x01 {
            return None;
        }
        if first.len() >= 6 && &first[1..6] == b"video" {
            let fourcc = slice4(first, 9)?;
            let (time_unit, spu) = common_timing(first);
            return Some(Self {
                media_type: MediaType::Video,
                codec_id: video_codec_id(&fourcc),
                tag: Some(CodecTag::fourcc(&fourcc)),
                width: read_u32(first, 45),
                height: read_u32(first, 49),
                channels: None,
                sample_rate: None,
                time_unit,
                spu,
            });
        }
        if first.len() >= 5 && &first[1..5] == b"text" {
            let (time_unit, spu) = common_timing(first);
            return Some(Self {
                media_type: MediaType::Subtitle,
                codec_id: CodecId::new("text"),
                tag: None,
                width: None,
                height: None,
                channels: None,
                sample_rate: None,
                time_unit,
                spu,
            });
        }
        if first.len() >= 6 && &first[1..6] == b"audio" {
            let wformat = hex_u16(&slice4(first, 9)?)?;
            let (time_unit, spu) = common_timing(first);
            return Some(Self {
                media_type: MediaType::Audio,
                codec_id: audio_codec_id(wformat),
                tag: Some(CodecTag::wave_format(wformat)),
                width: None,
                height: None,
                channels: read_u16(first, 45),
                sample_rate: rate_from_timing(time_unit, spu),
                time_unit,
                spu,
            });
        }
        None
    }

    /// One granule tick as `time_unit / (spu * 10_000_000)` seconds, or
    /// `None` when the timing fields are absent or unusable.
    pub fn time_base(&self) -> Option<TimeBase> {
        let num = i64::try_from(self.time_unit).ok()?;
        let den = i64::try_from(self.spu.checked_mul(10_000_000)?).ok()?;
        if num <= 0 || den <= 0 {
            return None;
        }
        Some(TimeBase::new(num, den))
    }

    /// Overwrite the stream parameters previously built from the
    /// (unknown) Xiph detection with the OGM-classified values.
    pub fn apply(&self, params: &mut CodecParameters) {
        params.media_type = self.media_type;
        params.codec_id = self.codec_id.clone();
        if self.width.is_some() {
            params.width = self.width;
        }
        if self.height.is_some() {
            params.height = self.height;
        }
        if self.channels.is_some() {
            params.channels = self.channels;
        }
        if self.sample_rate.is_some() {
            params.sample_rate = self.sample_rate;
        }
        if self.tag.is_some() {
            params.tag = self.tag.clone();
        }
    }
}

/// `(time_unit, spu)` from the common tail, both 0 when absent.
fn common_timing(first: &[u8]) -> (u64, u64) {
    (
        read_u64(first, 17).unwrap_or(0),
        read_u64(first, 25).unwrap_or(0),
    )
}

/// `spu * 10_000_000 / time_unit` samples (or frames) per second.
fn rate_from_timing(time_unit: u64, spu: u64) -> Option<u32> {
    if time_unit == 0 {
        return None;
    }
    let rate = spu.checked_mul(10_000_000)? / time_unit;
    u32::try_from(rate).ok().filter(|&rate| rate > 0)
}

/// Map an OGM video FourCC to a codec id, mirroring the common BMP-tag
/// names (`mpeg4`/`mjpeg`/`h264`); unlisted tags keep the FourCC text.
fn video_codec_id(fourcc: &[u8; 4]) -> CodecId {
    match fourcc {
        b"XVID" | b"DIVX" | b"DX50" | b"FMP4" | b"MP4V" => CodecId::new("mpeg4"),
        b"MJPG" | b"JPEG" => CodecId::new("mjpeg"),
        b"H264" | b"X264" | b"AVC1" | b"DAVC" => CodecId::new("h264"),
        other => CodecId::new(format!("ogm:{}", fourcc_label(other))),
    }
}

/// Map an OGM audio `wFormatTag` to a codec id; unlisted tags keep the
/// numeric tag.
fn audio_codec_id(wformat: u16) -> CodecId {
    match wformat {
        0x0001 => CodecId::new("pcm_s16le"),
        0x0055 => CodecId::new("mp3"),
        0x00FF => CodecId::new("aac"),
        0x2000 => CodecId::new("ac3"),
        0x674F | 0x6771 => CodecId::new("vorbis"),
        other => CodecId::new(format!("ogm:0x{other:04X}")),
    }
}

fn fourcc_label(fourcc: &[u8; 4]) -> String {
    if fourcc.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        fourcc.iter().map(|&b| b as char).collect()
    } else {
        fourcc.iter().map(|b| format!("{b:02X}")).collect()
    }
}

fn slice4(bytes: &[u8], off: usize) -> Option<[u8; 4]> {
    <[u8; 4]>::try_from(bytes.get(off..off + 4)?).ok()
}

fn hex_u16(bytes: &[u8]) -> Option<u16> {
    u16::from_str_radix(std::str::from_utf8(bytes).ok()?, 16).ok()
}

fn read_u16(bytes: &[u8], off: usize) -> Option<u16> {
    let b = bytes.get(off..off + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(bytes: &[u8], off: usize) -> Option<u32> {
    let b = bytes.get(off..off + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64(bytes: &[u8], off: usize) -> Option<u64> {
    let b = bytes.get(off..off + 8)?;
    Some(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First packet of the video stream of a real `.ogm` file: Xvid,
    /// 640 px wide, `time_unit = 417084`, `spu = 1` (≈23.976 fps).
    const VIDEO: &[u8] = &[
        0x01, b'v', b'i', b'd', b'e', b'o', 0x00, 0x00, 0x00, b'X', b'V', b'I', b'D', 0x38, 0x00,
        0x00, 0x00, 0x3C, 0x5D, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA4, 0x14, 0x02, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x80, 0x02, 0x00, 0x00, 0xE0, 0x01, 0x00, 0x00, 0x00,
    ];

    /// First packet of an OGM text (subtitle) stream.
    const TEXT: &[u8] = &[
        0x01, b't', b'e', b'x', b't', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x00,
        0x00, 0x00, 0x10, 0x27, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn parses_ogm_video_header() {
        let h = StreamHeader::parse(VIDEO).expect("video header");
        assert_eq!(h.media_type, MediaType::Video);
        assert_eq!(h.codec_id.as_str(), "mpeg4");
        assert_eq!(h.width, Some(640));
        assert_eq!(h.height, Some(480));
        assert!(matches!(h.tag, Some(CodecTag::Fourcc(_))));
        // 417084 / (1 * 10_000_000) seconds per frame.
        assert_eq!(h.time_base(), Some(TimeBase::new(417_084, 10_000_000)));
    }

    #[test]
    fn parses_ogm_text_as_subtitle() {
        let h = StreamHeader::parse(TEXT).expect("text header");
        assert_eq!(h.media_type, MediaType::Subtitle);
        assert_eq!(h.codec_id.as_str(), "text");
    }

    #[test]
    fn ignores_xiph_vorbis_header() {
        assert!(StreamHeader::parse(b"\x01vorbis\x00\x00\x00\x00\x02").is_none());
    }

    #[test]
    fn ignores_short_or_unrelated_packets() {
        assert!(StreamHeader::parse(b"").is_none());
        assert!(StreamHeader::parse(b"\x00video").is_none());
        assert!(StreamHeader::parse(b"\x01vid").is_none());
    }
}
