import glob
import random
import collections
import copy
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output


S3_addr = 'https://s3.us-east-2.amazonaws.com/n0lean-bucket/flower/'
REFOCUS_S3_ADDR = 'https://s3.amazonaws.com/photorefocus/'
refocus_img_list = [REFOCUS_S3_ADDR + str(i) + '.png' for i in range(100)]
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
]


with open('./eval_resultrefocus_final_4_index.json', 'r') as f:
    index = json.load(f)['index']

index = [[S3_addr + t for t in tri] for tri in index]
focusstack = collections.defaultdict(dict)
for ind in index:
    d = ind[0].split('/')[-1].split('_')[0]
    name = ind[0].split('/')[-1].split('.')[0].split('_')[-1]
    d = int(4 * (float(d) - float(name))) + 1
    focusstack[str(name)][d] = ind

focusstack = [val for key, val in focusstack.items()]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# heroku server
server = app.server


img_width = 256
img_height = 256
scale_factor = 1.3
gcounter = 0
slider1_loc = 1
slider0_loc = 0
slider2_loc = 1

fig2_links = ['test_raw.jpg', 'test_res_0.jpg', 'test_res_1.jpg', 'test_res_2.jpg', 'test_res_3.jpg']
fig2_links = [S3_addr + k for k in fig2_links]

latest_left = {
    'xaxis.range[0]': 0,
    'xaxis.range[1]': img_width,
    'yaxis.range[0]': 0,
    'yaxis.range[1]': img_height,
}


latest_mid = {
    'xaxis.range[0]': 0,
    'xaxis.range[1]': img_width,
    'yaxis.range[0]': 0,
    'yaxis.range[1]': img_height,
}


latest_right = {
    'xaxis.range[0]': 0,
    'xaxis.range[1]': img_width,
    'yaxis.range[0]': 0,
    'yaxis.range[1]': img_height,
}

latest_left0 = {
    'xaxis.range[0]': 0,
    'xaxis.range[1]': img_width,
    'yaxis.range[0]': 0,
    'yaxis.range[1]': img_height,
}


def get_random_image():
    return random.choice(focusstack)


fig0 = go.Layout(
    title="Refocus using light field",
    xaxis=go.layout.XAxis(
        visible=False,
        range=[0, img_width]),
    yaxis=go.layout.YAxis(
        visible=False,
        range=[0, img_height],
        scaleanchor='x'),
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
    images=[
        go.layout.Image(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=refocus_img_list[slider0_loc]
        )
    ]
)


fig1_stack = get_random_image()

fig1_left = go.Layout(
    title="Raw (Input)",
    xaxis=go.layout.XAxis(
        visible=False,
        range=[0, img_width]),
    yaxis=go.layout.YAxis(
        visible=False,
        range=[0, img_height],
        scaleanchor='x'),
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
    images=[
        go.layout.Image(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=fig1_stack[slider1_loc][0]
        )
    ]
)

fig1_mid = go.Layout(
    title="Ground Truth",
    xaxis=go.layout.XAxis(
        visible=False,
        range=[0, img_width]),
    yaxis=go.layout.YAxis(
        visible=False,
        range=[0, img_height],
        scaleanchor='x'),
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
    images=[
        go.layout.Image(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=fig1_stack[slider1_loc][0]
        )
    ]
)

fig1_right = go.Layout(
    title="Generated Sample",
    xaxis=go.layout.XAxis(
        visible=False,
        range=[0, img_width]),
    yaxis=go.layout.YAxis(
        visible=False,
        range=[0, img_height],
        scaleanchor='x'),
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
    images=[
        go.layout.Image(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=fig1_stack[slider1_loc][0]
        )
    ]
)


fig2_left = go.Layout(
    title="Generated Sample",
    xaxis=go.layout.XAxis(
        visible=False,
        range=[0, img_width]),
    yaxis=go.layout.YAxis(
        visible=False,
        range=[0, img_height],
        scaleanchor='x'),
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
    images=[
        go.layout.Image(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=fig2_links[0]
        )
    ]
)


fig2_right = go.Layout(
    title="Generated Sample",
    xaxis=go.layout.XAxis(
        visible=False,
        range=[0, img_width]),
    yaxis=go.layout.YAxis(
        visible=False,
        range=[0, img_height],
        scaleanchor='x'),
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
    images=[
        go.layout.Image(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=fig2_links[slider2_loc]
        )
    ]
)

latest_left2 = {
    'xaxis.range[0]': 0,
    'xaxis.range[1]': img_width,
    'yaxis.range[0]': 0,
    'yaxis.range[1]': img_height,
}

latest_right2 = {
    'xaxis.range[0]': 0,
    'xaxis.range[1]': img_width,
    'yaxis.range[0]': 0,
    'yaxis.range[1]': img_height,
}

with open('./intro.md', 'r') as f:
    intro_str = f.read()

with open('./method.md', 'r') as f:
    method_str = f.read()

with open('./visualize.md', 'r') as f:
    vis_str = f.read()

app.layout = html.Div(
    [
        # Title part
        html.Div(
            [
                html.H2(
                    ['Depth Field GAN'],
                    className="display-2"
                ),
                # html.Br(),
                html.H6(
                    'COMS 4995 Applied Deep Learning Project',
                    className="display-6"
                ),
                html.H6(
                    'By Pengyu Chen (pc2842), Xin Hu (xh2390)',
                    className="display-6"
                )
            ],
        ),
        html.Br(),

        # Introduction
        html.Div([
            html.Div(
                dcc.Markdown(intro_str, dangerously_allow_html=True),
            )
        ]),

        html.Div([
            html.Br(),
            html.Div([
                html.Div(className="col-6 col-md-4"),
                html.Div(
                    dcc.Graph(
                        id='fig0',
                        figure=go.Figure(
                            data=[{
                                'x': [0, img_width],
                                'y': [0, img_height],
                                'mode': 'markers',
                                'marker': {'opacity': 0},
                            }],
                            layout=fig0
                        )
                    ),
                    className="col-6 col-md-4"
                ),

            ], className='row align-items-center'),
        ]),

        html.Br(),
        html.Div(
            dcc.Slider(
                id='fig0_slider',
                min=0,
                max=99,
                value=0,
                marks={'0': 'near', '99': 'far'},
            ),
        ),
        html.Br(),
        html.Br(),


        html.Div([
            html.Br(),
            html.Button('Shuffle', id='fig1_button'),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([

                html.Div(
                    dcc.Graph(
                        id='fig1_left',
                        figure=go.Figure(
                            data=[{
                                'x': [0, img_width],
                                'y': [0, img_height],
                                'mode': 'markers',
                                'marker': {'opacity': 0},
                            }],
                            layout=fig1_left
                        )
                    ),
                    className="col-6 col-md-4"
                ),

                html.Div(
                    dcc.Graph(
                        id='fig1_mid',
                        figure=go.Figure(
                            data=[{
                                'x': [0, img_width],
                                'y': [0, img_height],
                                'mode': 'markers',
                                'marker': {'opacity': 0},
                            }],
                            layout=fig1_mid
                        )
                    ),
                    className="col-6 col-md-4"
                ),

                html.Div(
                    dcc.Graph(
                        id='fig1_right',
                        figure=go.Figure(
                            data=[{
                                'x': [0, img_width],
                                'y': [0, img_height],
                                'mode': 'markers',
                                'marker': {'opacity': 0}
                            }],
                            layout=fig1_right
                        )
                    ),
                    className="col-6 col-md-4"
                )
                ],
                # style={'columnCount': 2},
                className='row align-items-center'
            )
        ]),
        html.Br(),
        html.Div(
            dcc.Slider(
                id='fig1_slider',
                min=1,
                max=4,
                value=1,
                marks={str(k + 1): label for k, label in enumerate(['near', '', '', 'far'])},
                step=None
            ),
        ),
        html.Br(),
        html.Br(),

        # Method
        html.Div([
            html.Div(
                dcc.Markdown(method_str, dangerously_allow_html=True)
            )
        ]),

        # Visualize
        html.Div([
            html.Div([
                dcc.Markdown(vis_str, dangerously_allow_html=True),
            ]),

            html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.Div([

                    html.Div(
                        dcc.Graph(
                            id='fig2_left',
                            figure=go.Figure(
                                data=[{
                                    'x': [0, img_width],
                                    'y': [0, img_height],
                                    'mode': 'markers',
                                    'marker': {'opacity': 0},
                                }],
                                layout=fig2_left
                            )
                        ),
                        className="col-6 col-md-4"
                    ),

                    html.Div(
                        className="col-6 col-md-4"
                    ),

                    html.Div(
                        dcc.Graph(
                            id='fig2_right',
                            figure=go.Figure(
                                data=[{
                                    'x': [0, img_width],
                                    'y': [0, img_height],
                                    'mode': 'markers',
                                    'marker': {'opacity': 0}
                                }],
                                layout=fig2_right
                            )
                        ),
                        className="col-6 col-md-4"
                    )
                ],
                    # style={'columnCount': 2},
                    className='row align-items-center'
                )
            ]),
            html.Br(),
            html.Div(
                dcc.Slider(
                    id='fig2_slider',
                    min=1,
                    max=4,
                    value=1,
                    marks={str(k + 1): label for k, label in enumerate(['near', '', '', 'far'])},
                    step=None
                )
            ),
            html.Br(),
            html.Br(),


        ]),

    ],
    className="container",
    style={"margin": "0%", "margin-right": 'auto', 'margin-left': 'auto'},
)


@app.callback(
    [
        Output('fig1_left', 'figure'),
        Output('fig1_mid', 'figure'),
        Output('fig1_right', 'figure')
    ],
    [
        Input('fig1_button', 'n_clicks'),
        Input('fig1_left', 'relayoutData'),
        Input('fig1_mid', 'relayoutData'),
        Input('fig1_right', 'relayoutData'),
        Input('fig1_slider', 'value')
    ]
)
def random_fig1(counter, left, mid, right, slider):
    # if left['xaxis.range[0]'] == mid['xaxis.range[0]']:
    #     now = right
    # elif left['xaxis.range[0]'] == right['xaxis.range[0]']:
    #     now = mid
    # else:
    #     now = left
    global fig1_stack, latest_left, latest_mid, latest_right, gcounter, slider1_loc

    if slider is not None and slider1_loc != slider:
        slider1_loc = slider

    if slider is None:
        slider = slider1_loc

    if counter is not None and counter > gcounter:
        fig1_stack = get_random_image()
        gcounter = counter

    left = latest_left if left is None else left
    mid = latest_mid if mid is None else mid
    right = latest_right if right is None else right
    if 'xaxis.autorange' in left or 'xaxis.autorange' in mid or 'xaxis.autorange' in right:
        now = {
            'xaxis.range[0]': 0,
            'xaxis.range[1]': img_width,
            'yaxis.range[0]': 0,
            'yaxis.range[1]': img_height,
        }
    else:
        if int(left['xaxis.range[0]']) != int(latest_left['xaxis.range[0]']):
            now = left
        elif int(latest_right['xaxis.range[0]']) != int(right['xaxis.range[0]']):
            now = right
        else:
            now = mid

    now = {key: int(val) for key, val in now.items()}
    latest_left = copy.deepcopy(now)
    latest_right = copy.deepcopy(now)
    latest_mid = copy.deepcopy(now)
    slider1_loc = slider

    fig1_left = go.Layout(
        title="Raw",
        xaxis=go.layout.XAxis(
            visible=False,
            range=[now['xaxis.range[0]'], now['xaxis.range[1]']]),
        yaxis=go.layout.YAxis(
            visible=False,
            range=[now['yaxis.range[0]'], now['yaxis.range[1]']],
            scaleanchor='x'),
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
        images=[
            go.layout.Image(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fig1_stack[slider][0]
            )
        ]
    )

    fig1_mid = go.Layout(
        title="Ground Truth",
        xaxis=go.layout.XAxis(
            visible=False,
            range=[now['xaxis.range[0]'], now['xaxis.range[1]']]),
        yaxis=go.layout.YAxis(
            visible=False,
            range=[now['yaxis.range[0]'], now['yaxis.range[1]']],
            scaleanchor='x'),
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
        images=[
            go.layout.Image(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fig1_stack[slider][1]
            )
        ]
    )

    fig1_right = go.Layout(
        title="Generated Sample",
        xaxis=go.layout.XAxis(
            visible=False,
            range=[now['xaxis.range[0]'], now['xaxis.range[1]']]),
        yaxis=go.layout.YAxis(
            visible=False,
            range=[now['yaxis.range[0]'], now['yaxis.range[1]']],
            scaleanchor='x'),
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
        images=[
            go.layout.Image(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fig1_stack[slider][2]
            )
        ]
    )
    return [
        {
            'layout': fig1_left,
        },
        {
            'layout': fig1_mid,
        },
        {
            'layout': fig1_right,
        },
    ]


@app.callback(
    [
        Output('fig2_left', 'figure'),
        Output('fig2_right', 'figure')
    ],
    [
        Input('fig2_left', 'relayoutData'),
        Input('fig2_right', 'relayoutData'),
        Input('fig2_slider', 'value')
    ]
)
def random_fig1(left, right, slider):
    # if left['xaxis.range[0]'] == mid['xaxis.range[0]']:
    #     now = right
    # elif left['xaxis.range[0]'] == right['xaxis.range[0]']:
    #     now = mid
    # else:
    #     now = left
    global latest_left2, latest_right2, slider2_loc

    if slider is not None and slider2_loc != slider:
        slider2_loc = slider

    if slider is None:
        slider = slider2_loc

    left = latest_left2 if left is None else left
    right = latest_right2 if right is None else right
    if 'xaxis.autorange' in left or 'xaxis.autorange' in right:
        now = {
            'xaxis.range[0]': 0,
            'xaxis.range[1]': img_width,
            'yaxis.range[0]': 0,
            'yaxis.range[1]': img_height,
        }
    else:
        if int(left['xaxis.range[0]']) != int(latest_left2['xaxis.range[0]']):
            now = left
        else:
            now = right

    now = {key: int(val) for key, val in now.items()}
    latest_left2 = copy.deepcopy(now)
    latest_right2 = copy.deepcopy(now)
    slider2_loc = slider

    fig2_left = go.Layout(
        title="Raw",
        xaxis=go.layout.XAxis(
            visible=False,
            range=[now['xaxis.range[0]'], now['xaxis.range[1]']]),
        yaxis=go.layout.YAxis(
            visible=False,
            range=[now['yaxis.range[0]'], now['yaxis.range[1]']],
            scaleanchor='x'),
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
        images=[
            go.layout.Image(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fig2_links[0]
            )
        ]
    )

    fig2_right = go.Layout(
        title="Generated Sample",
        xaxis=go.layout.XAxis(
            visible=False,
            range=[now['xaxis.range[0]'], now['xaxis.range[1]']]),
        yaxis=go.layout.YAxis(
            visible=False,
            range=[now['yaxis.range[0]'], now['yaxis.range[1]']],
            scaleanchor='x'),
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
        images=[
            go.layout.Image(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fig2_links[slider]
            )
        ]
    )
    return [
        {
            'layout': fig2_left,
        },
        {
            'layout': fig2_right,
        },
    ]


@app.callback(
    [
        Output('fig0', 'figure'),
    ],
    [
        Input('fig0', 'relayoutData'),
        Input('fig0_slider', 'value')
    ]
)
def refocus(left, slider):
    global latest_left0, slider0_loc

    if slider is not None and slider0_loc != slider:
        slider0_loc = slider

    if slider is None:
        slider = slider0_loc

    left = latest_left0 if left is None else left
    if 'xaxis.autorange' in left:
        now = {
            'xaxis.range[0]': 0,
            'xaxis.range[1]': img_width,
            'yaxis.range[0]': 0,
            'yaxis.range[1]': img_height,
        }
    else:
        now = left

    now = {key: int(val) for key, val in now.items()}
    latest_left0 = copy.deepcopy(now)
    slider0_loc = slider

    fig0 = go.Layout(
        title="Refocus using light field",
        xaxis=go.layout.XAxis(
            visible=False,
            range=[now['xaxis.range[0]'], now['xaxis.range[1]']]),
        yaxis=go.layout.YAxis(
            visible=False,
            range=[now['yaxis.range[0]'], now['yaxis.range[1]']],
            scaleanchor='x'),
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 5},
        images=[
            go.layout.Image(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=refocus_img_list[slider0_loc]
            )
        ]
    )
    return [{'layout': fig0}]


if __name__ == '__main__':
    app.run_server()
