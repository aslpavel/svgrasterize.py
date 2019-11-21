## Python SVG rasterizer
This is very simple python+numpy SVG rasterize. It is defentitely incomplite and does have bugs.

### Usage
It can be used as stand alone executable to generate PNG out of SVG file.
```
usage: svgrasterize.py [-h] [-bg BG] [-fg FG] [-w WIDTH] [-id ID]
                       [-t TRANSFORM] [--linear-rgb]
                       [--fonts [FONTS [FONTS ...]]]
                       svg output

positional arguments:
  svg                   input SVG file
  output                output PNG file

optional arguments:
  -h, --help            show this help message and exit
  -bg BG                set default background color
  -fg FG                set default foreground color
  -w WIDTH, --width WIDTH
                        output width
  -id ID                render single element with specified `id`
  -t TRANSFORM, --transform TRANSFORM
                        apply additional transformation
  --linear-rgb          use linear RGB for rendering
  --fonts [FONTS [FONTS ...]]
                        paths to SVG files containing all fonts
```

### Supported features:
- [x] anti-aliased fill/stroke path
   - [x] with a color
   - [x] with linear and radial gradients
   - [x] with a pattern
- [x] masking and clipping
- [x] filters
   - [x] feColorMatrix
   - [x] feComposite
   - [x] feGaussianBlur (scipy is required)
   - [x] feMerge
   - [x] feMorphology
   - [x] feOffset
- [x] SVG fonts
- [x] text rendering
   - [x] text element
   - [x] tspan

### Demo
**Text**

![prompt](/demo/prompt.png)

**Some Icons**

![icons](/demo/icons.png)

**Material design icons**

![material design icons](/demo/material-design.png)
