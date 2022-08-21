from youtube_dl import YoutubeDL

audio_downloder = YoutubeDL({'format':'bestaudio'})

url = "https://music.youtube.com/watch?v=khnokW3Mw24&list=RDAMVM3_g2un5M350"


audio_downloder.extract_info(url)
