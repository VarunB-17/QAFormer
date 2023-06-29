<h1 align="center">
  <br>
  QAFormer
  <br>
</h1>

<h4 align="center"> An extractive question-answering model for SQuAD 1.1</h4>




![screenshot](https://github.com/VarunFuego17/thesisqat/blob/main/qaf.png)

## Features

* This model was built for my bachelor's thesis for the study of Artificial Intelligence at Vrije Universiteit Amsterdam.
* The QAFormer is based on QAnet that was published by Google in 2018.
* Training about around ~9 hours for 3 epochs.
* Check out the config.py files to modify hyperparameters to what your machine can handle.

## How To Use


```bash
# Clone this repository
$ git clone https://github.com/VarunFuego17/thesisqat.git

# Install dependencies (latest versions)
$ pip install pytorch 
$ pip install pyarrow
$ pip install wandb
$ pip install torchtext
$ pip install datasets
$ pip install spacy
$ pip install pandas
$ pip install numpy

# Go into the repository
$ cd dataloader
# Run the following file
$ python3 dataloader.py

# This should create the following files in the dataloader folder:
```
<img width="419" alt="image" src="https://github.com/VarunFuego17/thesisqat/assets/45126763/997d5ccd-c820-415c-a911-495923ca2404">

```bash
# Go into the repository
$ cd model
# Run the following command for creating the model:
$ python3 train.py --debug=1
# This should create the file -> "qaf_128_8_4"
# Run the following command for testing the model on the created dataset:
$ python3 train.py --debug=2
# Run the following command if you want to see if any errors appear:
$ python3 train.py --debug=0


```

> **Note**
> If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use `node` from the command prompt.


## Download

You can [download](https://github.com/amitmerchant1990/electron-markdownify/releases/tag/v1.2.0) the latest installable version of Markdownify for Windows, macOS and Linux.

## Emailware

Markdownify is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <bullredeyes@gmail.com> about anything you'd want to say about this software. I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/)

## Related

[markdownify-web](https://github.com/amitmerchant1990/markdownify-web) - Web version of Markdownify

## Support

<a href="https://www.buymeacoffee.com/5Zn8Xh3l9" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<p>Or</p> 

<a href="https://www.patreon.com/amitmerchant">
	<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a>

## You may also like...

- [Pomolectron](https://github.com/amitmerchant1990/pomolectron) - A pomodoro app
- [Correo](https://github.com/amitmerchant1990/correo) - A menubar/taskbar Gmail App for Windows and macOS

## License

MIT

---

> [amitmerchant.com](https://www.amitmerchant.com) &nbsp;&middot;&nbsp;
> GitHub [@amitmerchant1990](https://github.com/amitmerchant1990) &nbsp;&middot;&nbsp;
> Twitter [@amit_merchant](https://twitter.com/amit_merchant)

