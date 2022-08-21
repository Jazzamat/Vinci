from youtube_dl import YoutubeDL

class YoutubeConverter:

    @staticmethod
    def convert(url):
      
        audio_downloder = YoutubeDL({'format':'bestaudio'})
        audio_downloder.extract_info(url)
