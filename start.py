import ipywidgets as ipw


def get_start_widget(appbase, jupbase):
    # http://fontawesome.io/icons/
    template = """
    <table>
    <tr>

    <td valign="top"><ul>
    <li><a href="{appbase}/submit.ipynb" target="_blank">Submit calculation</a>
    <li><a href="{appbase}/search.ipynb" target="_blank">Search database</a>
    </ul></td>

    </tr>
    </table>
"""

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)


# EOF
