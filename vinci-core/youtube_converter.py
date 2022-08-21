# from youtube_dl import YoutubeDL
import yt_dlp

class YoutubeConverter:

    @staticmethod
    def convert(url):
      
        audio_downloder = yt_dlp.YoutubeDL({'format':'bestaudio',
        'noplaylist':True
        
        })
        audio_downloder.extract_info(url)