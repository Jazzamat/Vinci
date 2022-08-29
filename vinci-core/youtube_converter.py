# from youtube_dl import YoutubeDL
import yt_dlp

class YoutubeConverter:

    @staticmethod
    def convert(url):
      
        audio_downloder = yt_dlp.YoutubeDL({'format':'bestaudio',
        'postprocessor_args':'--audio-format wav',
        'noplaylist':True
        
        })
        audio_downloder.extract_info(url)

class YtDlp:


    @staticmethod
    def convert(url):
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'noplaylist':True,
            # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
            'postprocessors': [{  # Extract audio using ffmpeg
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)