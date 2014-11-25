
if [ -f embedded_cytoscape.hpp ]; then
  cp embedded_cytoscape.hpp embedded_cytoscape.hpp.bak
fi

echo "#pragma once" > embedded_cytoscape.hpp
echo "namespace Clustering {" >> embedded_cytoscape.hpp
echo "namespace Network {" >> embedded_cytoscape.hpp
echo "unsigned char viewer_header[] = {" >> embedded_cytoscape.hpp

xxd -i header.html | sed s/unsigned.*// >> embedded_cytoscape.hpp

echo 'const char* viewer_footer = "}); }); </script> </body> </html>";' >> embedded_cytoscape.hpp

echo '} // end namespace Network' >> embedded_cytoscape.hpp
echo '} // end namespace Clustering' >> embedded_cytoscape.hpp

